# OCR Robustness on Scanned Admission Forms (VISON)

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
<img alt="Tesseract OCR" src="https://img.shields.io/badge/Tesseract_OCR-5A5A5A?style=for-the-badge&logo=google&logoColor=white">
<img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
<img alt="Pillow" src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img alt="pandas" src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white">
<img alt="Levenshtein" src="https://img.shields.io/badge/python--Levenshtein-4B8BBE?style=for-the-badge&logo=python&logoColor=white">
<img alt="Mockaroo" src="https://img.shields.io/badge/Mockaroo-2E8B57?style=for-the-badge">
</p>

A computer-vision course project (DLSU, VISON) that measures how well **Tesseract OCR** reads data from scanned paper forms when the scan quality is poor, and tests whether simple image pre-processing can recover accuracy.

## Objective

Real-world form scans are rarely clean: they come in too bright, tinted, tilted, or grainy. The project set out to answer:

1. **How much does each type of scan defect hurt OCR accuracy?**
2. **Can classic OpenCV pre-processing (denoising, deskewing) bring accuracy back?**

The test case is the DLSU Office of Admissions **"Expression of Interest for Admission" form (Form No. UG-09)**. The target field read back from each scan is the applicant's **Last Name**, which is checked against the known ground truth.

## Dataset

| Item | Description |
|---|---|
| [MOCKAROO_DATA.csv](MOCKAROO_DATA.csv) | 200 fake applicants made with Mockaroo: reference no., last/first/middle name, date, and Philippine senior high school. This is the ground truth. |
| [Mail Merged Forms with Effects/](Mail%20Merged%20Forms%20with%20Effects/) | ~800 form images built by mail-merging the CSV into Form UG-09, then applying one of four defects (200 each): `bright`, `yellow` (tint), `rotate`, `noise`. |
| [Edited-9999/](Edited-9999/) | Another set of the same 800 forms (200 per defect), apparently re-edited versions. |
| [noise/](noise/) | 16 noisy samples (noise 185–200) used for single-image denoising tests. |
| [Mail Merged Forms - Rotated/](Mail%20Merged%20Forms%20-%20Rotated/) | 10 rotated samples used for deskew tests. |
| `drive-download-…zip` | Original Google Drive export of early rotated/bright samples. |

File naming: `form page 1-<effect>-<index>.jpg`, where `<index>` (001–200) is the matching row in the CSV.

## Stack

- **Python 3** in **Jupyter notebooks**
- **Tesseract OCR** via `pytesseract` (installed with Homebrew on macOS)
- **OpenCV** (`cv2`): denoising (`fastNlMeansDenoising`), Otsu thresholding, `minAreaRect` deskewing
- **Pillow (PIL)**: loading and cropping images
- **pandas**: reading the ground-truth CSV
- **python-Levenshtein**: edit distance for error rates
- **NumPy**, **imutils**, **matplotlib**
- **Mockaroo** (synthetic data) + mail merge (form generation)

## Method

1. Load each form image and crop to the form's data area (`y = 370–1140`).
2. Run `pytesseract.image_to_string` on the crop.
3. Check whether the label `"Last Name"` was detected (counted as a **match**). If so, take the word(s) after it as the OCR'd last name.
4. Compare against the CSV using Levenshtein distance:
   - **CER** (character error rate) = character edits ÷ length of true last name
   - **WER** (word error rate) = word edits ÷ number of words in true last name
5. Repeat for all 200 images per defect type, with and without pre-processing.

## Notebooks

| Notebook | Purpose |
|---|---|
| [Pytesseract.ipynb](Pytesseract.ipynb) / [Pytesseract (copy).ipynb](Pytesseract%20(copy).ipynb) | First experiments: Tesseract on government forms (e.g. Ontario Licence Appeal Tribunal) with hand-set bounding boxes. |
| [initial results .ipynb](initial%20results%20.ipynb) | Main experiment: baseline OCR for all four defects, then denoising on the noise set. |
| [Update_Notebook.ipynb](Update_Notebook.ipynb) | Follow-up: deskewing rotated forms before OCR. |

## Results

Baseline (no pre-processing), 200 images per defect:

| Defect | "Last Name" found | Missed | Avg CER | Avg WER |
|---|---|---|---|---|
| Brightness | 190 | 10 | 0.0023 | 0.010 |
| Yellow tint | **200** | 0 | 0.0020 | 0.015 |
| Rotation | 14 | **186** | 0.0202 | 0.020 |
| Noise | 12 | **188** | 0.0100 | 0.010 |

With pre-processing:

| Defect | Fix | "Last Name" found | Missed | Avg CER | Avg WER |
|---|---|---|---|---|---|
| Noise | OpenCV non-local means denoising (h=30) | **118** (from 12) | 82 | 0.170 | 0.185 |
| Rotation | `minAreaRect` deskew | Single-image test only; output was garbled/mirrored text, so the angle correction is not yet working | – | – | – |

### Takeaways

- **Brightness and tint barely affect Tesseract.** Nearly every form was read, with close to zero errors.
- **Rotation and noise break it.** Over 90% of those forms failed before any field could be found.
- **Denoising helps a lot.** It took the noisy set from 12 to 118 readable forms (about 10×), though the names it recovered have more character errors (~17% CER).
- **Deskewing still needs work.** The angle logic in `deskew()` produces a wrongly oriented image; fixing it is the clear next step for the rotated set.

### Caveats on the numbers

- Average CER/WER is divided by all 200 images, but forms where "Last Name" was not found count as 0 error. So low averages for rotation/noise come from few matches, not good reads. Read the match count first.
- In the baseline noise loop, `compute_error` is called once before the OCR text for that image exists (it reuses the previous image's text). The result is overwritten right after, so the totals are fine, but the line should be removed.
- The bounding box is fixed, so it only fits upright forms.
