from pathlib import Path
import os

import moviepy.video.io.ImageSequenceClip


if __name__ == "__main__":
    root_pth = Path(
        "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/viz"
    )
    predictions_pth = (
        root_pth
        / "raileniumad/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/railad/test/anomalous"
    )
    fps = 3
    anomalies_predictions = sorted(
        [
            pred_pth.as_posix()
            for pred_pth in list(predictions_pth.iterdir())
            if "amp" in str(pred_pth)
        ]
    )
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        anomalies_predictions, fps=fps
    )
    clip.write_videofile(
        "/home/mmhamdi/workspace/unsupervised/Unsupervised-Anomlay-Detection/RailADer/videos/railader_04112024_0915.mp4"
    )
