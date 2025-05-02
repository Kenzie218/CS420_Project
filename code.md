
# AIS Spam Detection Results

## Summary

Through testing, it was discovered that **radius does not significantly affect accuracy**. Instead, the **number of receptors (detectors)** is the primary factor influencing model accuracy.

## CSV Files

There are 4 CSV files used to display the results:

- Files starting with `AIS_` show results **after discovering that radius doesn't matter**.
- `radius_doesnt_matter.csv` demonstrates that **changing the radius has no real impact** on accuracy.

## Python Plotting Scripts

There are 4 Python programs used to visualize the results:

- **HeatMap.py** – Displays a heatmap of accuracy as the number of detectors changes.
- **Plot.py** – Line graph showing accuracy at each interval of `num_detectors`.
- **Plot1.py** – Shows that **radius has no effect** on accuracy.
- **Accuracy.py** – Displays **total accuracy** of the AIS model for each `num_detectors` value.

## Irrelevant Results

The `irrelevant_results/` folder contains output from earlier, more intensive runs **before discovering that radius does not impact accuracy**.

## Note on Randomness

Initially, results appeared inconsistent and seemed random. Upon inspection of the AIS model source code, it was found that it used **random seeds** to generate results. This randomness meant one run might be good at detecting spam while the next could be poor.

After this discovery, the seed was fixed to a constant value. This change **produced consistent and reliable data**, allowing a more accurate assessment of how different parameters affected model performance.
