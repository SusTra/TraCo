# Complementing traditional Traffic Counter (TraCo) sources with emergent solutions

 This repository is supplementing the paper **Citizen science for traffic monitoring: Investigating the potentials for complementing traditional counters with crowdsourced data**.


## Main files
The main files that can be used to reproduce the results reported in the paper are as follows
* [`get_telraam_data.ipynb`](get_telraam_data.ipynb): get counters data through the Telraam API.
* [`get_weather_data.ipynb`](get_weather_data.ipynb): get weather through the visual-crossing-weather API.
* [`telraam_preprocess.ipynb`](telraam_preprocess.ipynb): preprocess the Telraam data.
* [`telraam_basic.ipynb`](telraam_basic.ipynb): basic analyses and visualisation of the Telraam data.
* [`counters_ILC_basic.ipynb`](counters_ILC_basic.ipynb): basic analyses and visualisation of the ILC counters data.
* [`counters_concat.ipynb`](counters_concat.ipynb): merge the Telraam data with inductive loop counters data and weather data per a specific segment.
* [`regression.ipynb`](regression.ipynb): perform the regression analysis using different regression models.
* [`analyse_regression_results.ipynb`](analyse_regression_results.ipynb): analyse and visualise the regression results.
* [`summarize_best_models.ipynb`](summarize_best_models.ipynb): summary of the best models.
* [`best_models.pdf`](best_models.pdf): description of the best models obtained in our analysis.
* [`best_models.pickle`](best_models.pickle): implementations of the best models in a `pickle` format; can be loaded by importing `pickle` and using the following code:
```
with open('best_models.pickle', 'rb') as handle:
    best_models = pickle.load(handle)
```

## Additional folders
* [`data`](/data/): data used in the analysis.
* [`figs`](/figs/): figures produced during the analysis.


## Requirements
You should install the following libraries before using this code 
* `SciPy`
* `NumPy`
* `seaborn`
* `matplotlib`
* `pandas`
* `scikit-learn`
* `statsmodels`

## How to cite this work
Please cite this work as *Janež M., Verovšek Š., Zupančič T., Moškon M., "Citizen science for traffic monitoring: Investigating the potentials for complementing traffic counters with crowdsourced data." Sustainability 2022,14, 622. https://doi.org/10.3390/su14020622*