# CellSpot

Batteries hidden in thrown-away devices, lithium-ion above all, start fires when garbage trucks compact them and when
e-waste shredders tear them open. CellSpot looks for them at two points:

- **Before the bin.** A resident photographs an item. A model on the phone names the device and, since every device
  type it knows has a battery, says to keep it out of the trash and recycling bins and take it to a drop-off. When the
  model is unsure, it asks one question ("Does it charge, light up, make a sound or have a screen?") instead of
  guessing.
- **Before the shredder.** At an e-waste plant, an X-ray of each device shows whether a battery is still inside. The
  detector marks batteries and empty battery compartments, and devices with a battery are pulled for removal.

Both models run in the browser with [onnxruntime-web](https://onnxruntime.ai/); photos and X-rays are not uploaded.
Built for Arbiter Hacks V1, Theme track "Rethinking Waste".

## Results on devices the models never saw

| Check | Test set | Result |
|---|---|---|
| X-ray: is a battery inside? | 482 X-rays of 91 devices | 294 of 304 with a battery flagged (97%), 10 missed, 26 flagged without one (score threshold 0.1) |
| X-ray: battery boxes | same | AP50 0.82; 0.81 on the 91 high-quality X-rays |
| Photo: naming the device | 91 photos of 91 devices | 80% named correctly, 90% right in the top two |
| Photo: the app's rule | same | named 65 (8 with a wrong name, still the right advice), asked the question on 26 |

XBAT+ splits its devices into training and test sets, so no test device appears in training. The X-ray test set has
482 images of 91 devices; the photo test set has one photo of each of the 91 devices. Some device types have only two
to four test photos, so their per-type shares move a lot with a single photo.

## How it works

- `training/train_classifier.py`: MobileNetV3-Large (ImageNet weights) fine-tuned on crops around the device in
  XBAT+ webcam photos, 15 device types. Learning rate, input size and class balancing were chosen by 5-fold
  cross-validation on the training devices (folds split by device), and the number of epochs on a device-disjoint
  validation slice. The app names the device at 60% confidence or more and asks the question otherwise.
- `training/train_detector.py`: Faster R-CNN MobileNetV3-Large FPN (COCO weights) fine-tuned on XBAT+ X-rays,
  17 classes: the 15 device types, `Battery` and `NoBattery` (an empty battery compartment). The score threshold for
  pulling a device is chosen on validation devices for at least 90% precision.
- `training/export_onnx.py`: ONNX export, checked against PyTorch with onnxruntime.
- `web/`: a static site (no build step, no server code). `web/samples` holds a few XBAT+ test photos and X-rays.

## Run it

```sh
cd web
python3 -m http.server 8000
```

Then open http://localhost:8000. The models are in `web/models`.

## Train it

The data is XBAT+ 1.0 (resized archives) from [Zenodo record 18022530](https://zenodo.org/records/18022530). Unzip
the `res_*` archives into `../cellspot-data/xbat` (or set `XBAT_DIR`), then:

```sh
cd training
python train_classifier.py --out ../runs/classifier
python train_detector.py --out ../runs/detector
python export_onnx.py ../runs/classifier classifier
python export_onnx.py ../runs/detector detector
python make_web_assets.py ../runs/classifier ../runs/detector
```

The release ships labels without class names; `training/common.py` records the order, checked against the images
and the per-class counts.

## Limits

- The photo model knows 15 device types, all with batteries. It cannot tell that an item has no battery; for items it
  does not recognise, the question decides. We tried to catch unfamiliar items by measuring how far a photo's embedding sits
  from the training photos (nearest photo and class means, on 960- and 1280-d features), but X-ray images, a very different
  kind of picture, still passed 60-91% of the time, so the app relies on its confidence threshold and the question.
- All X-rays come from one scanner (NTB EZ-240) and all photos from one webcam setup. Other scanners and phone photos
  will differ; a deployment would need images from its own line to check the numbers.
- CellSpot gives disposal advice from the US EPA's general guidance. Local rules vary.

## Credits

- XBAT+ 1.0 datasets: Rukundo, O., Khan, R., Grua, E.M. et al., "Annotated datasets for waste electrical and
  electronic equipment identification and battery detection and classification", Scientific Data (2026),
  doi:10.1038/s41597-026-07606-4. CC BY 4.0. The sample images in `web/samples` come from its test sets.
- [onnxruntime-web](https://github.com/microsoft/onnxruntime) 1.30.0, MIT licence (`web/vendor/ort`).
- torchvision model definitions and pretrained weights (BSD-3-Clause code; ImageNet and COCO weights).
- Disposal guidance: [US EPA, Used Lithium-Ion Batteries](https://www.epa.gov/recycle/used-lithium-ion-batteries).
- Built with help from an AI coding assistant (Anthropic Claude).

## Licence

MIT for the code in this repository. The XBAT+ sample images remain under CC BY 4.0.
