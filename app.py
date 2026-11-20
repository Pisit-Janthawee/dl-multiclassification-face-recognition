"""Face verification demo (Gradio): two photos in → "same person" or "different / unknown" out.

The face is found and cropped with MTCNN, turned into a 512-number embedding by FaceNet (VGGFace2 weights),
and the two embeddings are compared by cosine similarity against the threshold chosen in the notebook.

    pip install -r requirements.txt
    pip install --no-deps facenet-pytorch
    python app.py

Optional environment variables
    FACE_SETTINGS   settings.json written by the notebook (default models/settings.json; otherwise the default threshold below)
    LFW_DIR         LFW person folders → a few public example pairs in the app (no faces are stored in this repository)

A demo on public data, not an identity check: do not use it on people who have not agreed.
"""
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = Path(os.environ.get('FACE_SETTINGS', HERE / 'models' / 'settings.json'))
SETTINGS = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
THRESHOLD = float(SETTINGS.get('threshold', 0.53))          # notebook value at FAR 0.1 % (LFW, unseen people)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DETECTOR = MTCNN(image_size=160, margin=20, post_process=True, device=DEVICE)
MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def face_tensor(image):
    """Detected face (160 x 160, normalised like FaceNet expects) and a preview, or (None, None) if no face is found."""
    pil = Image.fromarray(np.asarray(image)).convert('RGB')
    face = DETECTOR(pil)
    if face is None:
        return None, None
    preview = ((face.permute(1, 2, 0).numpy() * 128 + 127.5).clip(0, 255)).astype(np.uint8)
    return face, preview


def verify(image_a, image_b):
    if image_a is None or image_b is None:
        return None, None, 'Upload two photos.'
    (fa, pa), (fb, pb) = face_tensor(image_a), face_tensor(image_b)
    if fa is None or fb is None:
        return pa, pb, 'No face found in ' + ('photo A' if fa is None else 'photo B') + '.'
    with torch.no_grad():
        e = F.normalize(MODEL(torch.stack([fa, fb]).to(DEVICE)), dim=1)
    s = float((e[0] * e[1]).sum())
    verdict = 'Same person' if s >= THRESHOLD else 'Different people (or unknown)'
    return pa, pb, f'{verdict}\ncosine similarity {s:.2f} · threshold {THRESHOLD:.2f} (false accept rate 0.1 % on unseen LFW people)'


def examples():
    root = os.environ.get('LFW_DIR')
    if not root or not Path(root).exists():
        return None
    people = [p for p in sorted(Path(root).iterdir()) if p.is_dir() and len(list(p.glob('*.jpg'))) >= 2][:40:10]
    pairs = [[str(sorted(p.glob('*.jpg'))[0]), str(sorted(p.glob('*.jpg'))[1])] for p in people[:2]]
    if len(people) >= 2:
        pairs.append([str(sorted(people[0].glob('*.jpg'))[0]), str(sorted(people[1].glob('*.jpg'))[0])])
    return pairs


with gr.Blocks(title='Face verification') as demo:
    gr.Markdown('## Face verification: same person or not?\nUpload two photos. The face is found, turned into an embedding, '
                'and the two embeddings are compared with the threshold chosen on people the model never saw. '
                'A demo on public data, not an identity check.')
    with gr.Row():
        a = gr.Image(label='photo A (enrolled)')
        b = gr.Image(label='photo B (new)')
    with gr.Row():
        fa = gr.Image(label='face found in A', height=160)
        fb = gr.Image(label='face found in B', height=160)
        result = gr.Textbox(label='result', lines=3)
    gr.Button('Compare', variant='primary').click(verify, [a, b], [fa, fb, result])
    ex = examples()
    if ex:
        gr.Examples(ex, [a, b])

if __name__ == '__main__':
    demo.launch()
