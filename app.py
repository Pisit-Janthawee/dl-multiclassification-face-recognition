"""Face recognition demo (Gradio) with my four friends as the enrolled people.

Tab "Who is this?"  → one photo in → Dong, Aum, TJ, Mark, or "unknown" out (gallery = one photo per friend in friends/)
Tab "Same person?"  → two photos in → "same person" or "different / unknown" out

The face is found and cropped with MTCNN, turned into a 512-number embedding by FaceNet (VGGFace2 weights),
and compared by cosine similarity against the threshold chosen in the notebook (false accept rate 0.1 % on unseen LFW people).

    pip install -r requirements.txt
    pip install --no-deps facenet-pytorch
    python app.py

Optional environment variables
    FACE_SETTINGS   settings.json written by the notebook (default models/settings.json; otherwise the default threshold below)
    FRIENDS_DIR     folder with the enrolled photos (default friends/, file name = person name in lower case)

A demo, not an identity check: only use photos of people who agreed.
"""
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = Path(os.environ.get('FACE_SETTINGS', HERE / 'models' / 'settings.json'))
SETTINGS = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
THRESHOLD = float(SETTINGS.get('threshold', 0.53))          # notebook value at FAR 0.1 % (LFW, unseen people)
FRIENDS_DIR = Path(os.environ.get('FRIENDS_DIR', HERE / 'friends'))
FRIENDS = ['Dong', 'Aum', 'TJ', 'Mark']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DETECTOR = MTCNN(image_size=160, margin=40, post_process=False, device=DEVICE)
MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)


def find_face(image):
    """Cropped face as a uint8 array (160 x 160 x 3), or None if no face is found."""
    photo = Image.fromarray(np.asarray(image)).convert('RGB')
    photo.thumbnail((1600, 1600))                    # large camera photos: faster, same face
    face = DETECTOR(photo)
    return None if face is None else face.permute(1, 2, 0).numpy().astype(np.uint8)


def embed(faces):
    x = torch.from_numpy((np.stack(faces).astype(np.float32) - 127.5) / 128).permute(0, 3, 1, 2)
    with torch.no_grad():
        return F.normalize(MODEL(x.to(DEVICE)), dim=1).cpu().numpy()


GALLERY_FACES = {n: find_face(ImageOps.exif_transpose(Image.open(FRIENDS_DIR / f'{n.lower()}.png'))) for n in FRIENDS if (FRIENDS_DIR / f'{n.lower()}.png').exists()}
GALLERY = embed(list(GALLERY_FACES.values())) if GALLERY_FACES else None


def identify(image):
    if image is None:
        return None, 'Upload a photo.', {}
    face = find_face(image)
    if face is None:
        return None, 'No face found.', {}
    sims = embed([face])[0] @ GALLERY.T
    scores = {n: float(max(s, 0)) for n, s in zip(GALLERY_FACES, sims)}
    best = int(sims.argmax())
    name = list(GALLERY_FACES)[best]
    verdict = name if sims[best] >= THRESHOLD else 'Unknown (no friend is close enough)'
    return face, f'{verdict}\nclosest: {name}, similarity {sims[best]:.2f} · threshold {THRESHOLD:.2f}', scores


def verify(image_a, image_b):
    if image_a is None or image_b is None:
        return None, None, 'Upload two photos.'
    fa, fb = find_face(image_a), find_face(image_b)
    if fa is None or fb is None:
        return fa, fb, 'No face found in ' + ('photo A' if fa is None else 'photo B') + '.'
    e = embed([fa, fb])
    s = float(e[0] @ e[1])
    verdict = 'Same person' if s >= THRESHOLD else 'Different people (or unknown)'
    return fa, fb, f'{verdict}\ncosine similarity {s:.2f} · threshold {THRESHOLD:.2f}'


with gr.Blocks(title='Face recognition') as demo:
    gr.Markdown('## Face recognition: my four friends, and "unknown" for everyone else\n'
                'The face is found, turned into an embedding, and compared with one enrolled photo per friend. '
                'The threshold was chosen on people the model never saw (false accept rate 0.1 %). A demo, not an identity check.')
    with gr.Tab('Who is this?'):
        with gr.Row():
            photo = gr.Image(label='photo')
            with gr.Column():
                face_out = gr.Image(label='face found', height=160)
                answer = gr.Textbox(label='answer', lines=2)
                scores = gr.Label(label='similarity to each friend', num_top_classes=4)
        gr.Button('Who is this?', variant='primary').click(identify, photo, [face_out, answer, scores])
        existing = [str(FRIENDS_DIR / f'{n.lower()}.png') for n in GALLERY_FACES]
        if existing:
            gr.Examples(existing, photo)
    with gr.Tab('Same person?'):
        with gr.Row():
            a = gr.Image(label='photo A (enrolled)')
            b = gr.Image(label='photo B (new)')
        with gr.Row():
            fa_out = gr.Image(label='face found in A', height=160)
            fb_out = gr.Image(label='face found in B', height=160)
            result = gr.Textbox(label='result', lines=2)
        gr.Button('Compare', variant='primary').click(verify, [a, b], [fa_out, fb_out, result])

if __name__ == '__main__':
    demo.launch()
