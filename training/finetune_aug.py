"""finetune.py는 그대로 두고, train 이미지에만 증강을 켜서 학습한다.

예 (E6): python -m training.finetune_aug --data dataset --epochs 15 --batch-size 8 --lr 5e-5 --output artifacts/beans_resnet18_aug_e15

finetune.py의 run()을 그대로 쓰고, 이미지 읽는 클래스(FolderImages)만 증강 버전으로 바꿔 끼운다.
- dataset/train → 증강 O (매번 다른 변형)
- dataset/val   → 증강 X (원본 그대로 → 모델 고르기 공정)
저장되는 preprocessor_config.json은 finetune.py와 같다 → 서빙(hf_model.py) 코드 변경 없음.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

from training import finetune
from training.augment import train_augment


class AugFolderImages(finetune.FolderImages):
    def __init__(self, root: Path, classes, processor):
        super().__init__(root, classes, processor)
        self.augment = root.name == "train"   # train 폴더일 때만 증강
        self.rng = random.Random(42)          # 증강 무작위도 seed 고정

    def __getitem__(self, index):
        if not self.augment:
            return super().__getitem__(index)
        path, label = self.records[index]
        with Image.open(path) as raw:
            image = train_augment(raw.convert("RGB"), self.rng)
            tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": finetune.torch.tensor(label, dtype=finetune.torch.long)}


AUG_NOTE = ("train only: 50% crop_in 0.4-1.0 / 30% zoom_out 0.5-0.9 on blurred bg / 20% none; "
            "hflip 0.5, vflip 0.5, rotate -30~30; brightness 0.7-1.3, contrast/saturation 0.8-1.2; each 30%: blur r0.5-2, lowres 0.3-0.7, gaussian noise sigma3-12, jpeg q30-80; seed 42")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="artifacts/beans_resnet18_aug_e15")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    finetune.FolderImages = AugFolderImages   # run() 안에서 쓰는 클래스를 증강 버전으로 교체
    finetune.run(args)

    # training_report.json에 증강 설정 기록
    report_path = Path(args.output) / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update({"augmentation": AUG_NOTE, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size})
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("증강 설정을 training_report.json에 기록했습니다.")
