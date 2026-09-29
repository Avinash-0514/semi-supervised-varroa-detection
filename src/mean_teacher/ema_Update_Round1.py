from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\User\Desktop\Thesis\Src_Yolo")
RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"

teacher0 = PROJECT_ROOT / "mean_teacher" / "teacher_round0.pt"
student1 = RUNS_ROOT / "yolo8_mean_teacher_round1" / "weights" / "best.pt"
teacher1 = PROJECT_ROOT / "mean_teacher" / "teacher_round1.pt"
EMA_DECAY = 0.99

@torch.no_grad()
def ema_update_teacher(teacher_pt, student_pt, out_teacher_pt, decay):
    t = YOLO(str(teacher_pt))
    s = YOLO(str(student_pt))
    t_state = t.model.state_dict()
    s_state = s.model.state_dict()

    for k in t_state.keys():
        if k in s_state and t_state[k].dtype.is_floating_point:
            t_state[k].mul_(decay).add_(s_state[k] * (1.0 - decay))

    t.model.load_state_dict(t_state, strict=False)
    t.save(str(out_teacher_pt))

def main():
    print("Making teacher_round1 via EMA...")
    ema_update_teacher(teacher0, student1, teacher1, EMA_DECAY)
    print("✅ Saved:", teacher1)

if __name__ == "__main__":
    main()
