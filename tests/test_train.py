import sys
from pathlib import Path

from pitchlens.training.dataset import inspect_dataset
from pitchlens.training.train import TrainSettings, build_train_args, suggest_imgsz, train

sys.path.insert(0, str(Path(__file__).parent))
from test_dataset import _make_dataset  # noqa: E402


class FakeYOLO:
    """Stands in for ultralytics.YOLO: records the call and writes a best.pt like real training."""
    calls: list = []

    def __init__(self, weights):
        self.weights = weights
        self.trainer = None

    def train(self, **kwargs):
        FakeYOLO.calls.append((self.weights, kwargs))
        best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"weights")
        self.trainer = type("T", (), {"best": str(best)})()


def test_imgsz_follows_the_ball_size(tmp_path):
    small = inspect_dataset(_make_dataset(tmp_path / "small"))        # 20 px ball on 1920 px images
    assert suggest_imgsz(small) == 1280                                # ~6.7 px at 640: too small
    big = _make_dataset(tmp_path / "big")
    for split in ("train", "valid"):
        for f in (big / split / "labels").glob("*.txt"):
            f.write_text("0 0.5 0.5 0.02 0.035\n")                    # ~38 px ball
    assert suggest_imgsz(inspect_dataset(big)) == 640


def test_train_args():
    args = build_train_args(Path("ds/data.yaml"), TrainSettings(epochs=5, device="0"), imgsz=1280)
    assert args["data"].endswith("data.yaml") and args["imgsz"] == 1280 and args["epochs"] == 5
    assert args["device"] == "0" and args["flipud"] == 0.0 and args["hsv_h"] == 0.0
    assert "device" not in build_train_args(Path("d"), TrainSettings(), 640)


def test_train_runs_and_copies_best_weights(tmp_path, capsys):
    FakeYOLO.calls.clear()
    ds = _make_dataset(tmp_path / "ds")
    settings = TrainSettings(epochs=3, project=str(tmp_path / "runs"), output=str(tmp_path / "models" / "football.pt"))
    out = train(ds, settings, yolo_cls=FakeYOLO)
    assert out == Path(settings.output) and out.read_bytes() == b"weights"
    weights, kwargs = FakeYOLO.calls[0]
    assert weights == "yolov8m.pt" and kwargs["imgsz"] == 1280 and kwargs["epochs"] == 3
    assert "model: " in capsys.readouterr().out


def test_dataset_problems_stop_training_unless_forced(tmp_path):
    FakeYOLO.calls.clear()
    ds = _make_dataset(tmp_path / "ds", splits=("train",))           # no validation split
    settings = TrainSettings(project=str(tmp_path / "runs"), output=str(tmp_path / "football.pt"))
    assert train(ds, settings, yolo_cls=FakeYOLO) is None and FakeYOLO.calls == []
    assert train(ds, settings, force=True, yolo_cls=FakeYOLO) is not None
