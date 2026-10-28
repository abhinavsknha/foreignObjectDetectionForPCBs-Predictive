# Training notebooks

These notebooks are the record of how the two detectors were trained. The application itself does not use them: it loads the finished weights (see the top-level README).

| Notebook | What it does | Runs on |
| :--- | :--- | :--- |
| `YOLO.ipynb` | Track 1. Trains YOLO26n on DsPCBSD+ (8,208 training and 2,051 validation images, configured by `pcb_data.yaml`). | Local or Colab |
| `YOLO_geminiUpdated.ipynb` | Same pipeline in sample mode: 5 epochs on about 50 images, to check the whole workflow quickly before a full run. | Local or Colab |
| `DETR_Track.ipynb` | Track 2. Sets up the dataset (auto-discovery, caching), converts YOLO labels to COCO JSON, applies the adaptive preprocessing, and trains RT-DETR. | Colab or local |
| `DETR_Track_Kaggle.ipynb` | Track 2 on a Kaggle GPU. Continues RT-DETR-L training from the stage-1 checkpoint, evaluates on the validation split, and exports ONNX. | Kaggle |

Notes:
* They were written for Colab and Kaggle, so paths such as `/content/drive` and `/kaggle/input` need adjusting to run anywhere else.
* The datasets and trained weights are not stored in the repository.
* Saved outputs are cleared so the files stay small and diff cleanly.
