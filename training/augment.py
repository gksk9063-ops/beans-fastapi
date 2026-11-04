"""거리·구도·빛 변화를 흉내 내는 PIL 변형 모음.

- train_augment: 학습(train) 이미지에만 무작위로 적용 (Val/Test에는 쓰지 않음)
- SHIFT_CONDITIONS: Val 사진을 고정된 방식으로 바꿔 "거리·구도 변화"에 강한지 비교할 때 사용
회전·뒤집기·확대·축소·밝기는 병의 종류(정답)를 바꾸지 않는다.
"""
import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _blurred_background(img: Image.Image) -> Image.Image:
    """원본을 아주 흐리게 만든 배경 (검은 여백 대신 자연스러운 배경)."""
    w, h = img.size
    return img.resize((max(1, w // 10), max(1, h // 10))).resize((w, h)).filter(ImageFilter.GaussianBlur(6))


def crop_in(img, ratio, cx=0.5, cy=0.5):
    """가까이 찍은 효과: ratio 크기만큼 잘라서 원래 크기로 키운다. (cx, cy) = 자를 중심 위치 0~1"""
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    left = min(max(0, int(cx * w - cw / 2)), w - cw)
    top = min(max(0, int(cy * h - ch / 2)), h - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def zoom_out(img, ratio, cx=0.5, cy=0.5):
    """멀리서 찍은 효과: 사진을 ratio 크기로 줄여 흐린 배경 위에 붙인다 (8일차 Cut-Paste와 같은 생각)."""
    w, h = img.size
    bg = _blurred_background(img)
    small = img.resize((int(w * ratio), int(h * ratio)), Image.BILINEAR)
    x = int((w - small.width) * cx)
    y = int((h - small.height) * cy)
    bg.paste(small, (x, y))
    return bg


def rotate(img, degrees):
    """구도 변화: 회전 후 생긴 빈 모서리는 흐린 배경으로 채운다."""
    rotated = img.rotate(degrees, resample=Image.BILINEAR)
    mask = Image.new("L", img.size, 255).rotate(degrees, resample=Image.BILINEAR)
    return Image.composite(rotated, _blurred_background(img), mask)


def color(img, brightness=1.0, contrast=1.0, saturation=1.0):
    """빛 변화: 1.0 = 그대로, 0.7 = 30% 어둡게/약하게"""
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return ImageEnhance.Color(img).enhance(saturation)


def blur(img, radius):
    """흔들림·초점 안 맞음: 가우시안 블러 (radius 클수록 흐림)"""
    return img.filter(ImageFilter.GaussianBlur(radius))


def low_res(img, ratio):
    """저화질·썸네일: 작게 줄였다가 다시 키운다"""
    w, h = img.size
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def noise(img, sigma, rng=None):
    """어두운 곳 카메라 노이즈: 픽셀마다 정규분포 N(0, sigma) 값을 더한다"""
    gen = np.random.default_rng(None if rng is None else rng.randrange(2**32))
    arr = np.asarray(img, dtype=np.float32) + gen.normal(0, sigma, (img.height, img.width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg(img, quality):
    """메신저·웹 압축: 낮은 품질 JPEG로 저장했다가 다시 읽는다"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def train_augment(img, rng: random.Random):
    """학습 사진 1장에 무작위 변형을 적용한다."""
    # 1) 거리: 50% 가까이(0.4~1.0만 남김), 30% 멀리(0.5~0.9로 축소), 20% 그대로
    r = rng.random()
    if r < 0.5:
        img = crop_in(img, rng.uniform(0.4, 1.0), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
    elif r < 0.8:
        img = zoom_out(img, rng.uniform(0.5, 0.9), rng.random(), rng.random())
    # 2) 구도: 좌우·상하 뒤집기, -30~30도 회전
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.5:
        img = ImageOps.flip(img)
    img = rotate(img, rng.uniform(-30, 30))
    # 3) 빛: 밝기 0.7~1.3, 대비·채도 0.8~1.2 (색조는 안 바꿈: 갈색/노란 병반 색이 정답 근거라서)
    img = color(img, rng.uniform(0.7, 1.3), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2))
    # 4) 화질 (각각 30%): 작은 병반 점이 지워지지 않게 약하게만
    if rng.random() < 0.3:
        img = blur(img, rng.uniform(0.5, 2.0))
    if rng.random() < 0.3:
        img = low_res(img, rng.uniform(0.3, 0.7))
    if rng.random() < 0.3:
        img = noise(img, rng.uniform(3, 12), rng)
    if rng.random() < 0.3:
        img = jpeg(img, rng.randint(30, 80))
    return img


# Val 비교용 고정 변형 (무작위 없음 → 모델끼리 같은 조건으로 비교)
SHIFT_CONDITIONS = {
    "original": lambda im: im,
    "closer_0.6": lambda im: crop_in(im, 0.6),            # 가운데 60%만
    "offcenter_0.7": lambda im: crop_in(im, 0.7, 0.3, 0.3),  # 왼쪽 위로 치우친 구도
    "farther_0.6": lambda im: zoom_out(im, 0.6),          # 60% 크기로 멀리
    "farther_0.4": lambda im: zoom_out(im, 0.4),          # 40% 크기 (학습 범위 0.5 밖)
    "rotate_45": lambda im: rotate(im, 45),               # 학습 범위 30도 밖
    "dark_0.6": lambda im: color(im, 0.6),                # 40% 어둡게 (학습 범위 0.7 밖)
    "blur_3": lambda im: blur(im, 3),                     # 학습 범위 2 밖
    "lowres_0.25": lambda im: low_res(im, 0.25),          # 125px 썸네일 수준
    "noise_15": lambda im: noise(im, 15, random.Random(0)),
    "jpeg_20": lambda im: jpeg(im, 20),
}
