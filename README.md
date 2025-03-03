# DeepRRTime: Robust Time-series Forecasting with a Regularized INR Basis (TMLR 2025)

<p align="center">
<img src=".\pics\deeprrtime.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall approach of DeepRRTime.
</p>

Official PyTorch code repository for the [DeepRRTime paper](https://openreview.net/forum?id=uDRzORdPT7). DeepRRTime advances state-of-the-art in time-series forecasting amongst deep time-index models, a recent modeling paradigm for time-series forecasting.

## Requirements

Dependencies for this project can be installed by:

```bash
pip install -r requirements.txt
```

## Experiments

Steps to reproduce the results in Tables 1 and 2:
1. Download datasets
   * Pre-processed datasets can be downloaded from the following
  links, [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/)
  or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing), as obtained
  from [Autoformer's](https://github.com/thuml/Autoformer) GitHub repository.
   * Place the downloaded datasets into the `storage/datasets/` folder, e.g. `storage/datasets/ETT-small/ETTm2.csv`.

2. Generate experiments for various combinations of forecast-horizons (e.g., 96, 192, 336 or 720), lookback multipliers (e.g., 1, 3, 5, 7 or 9) and regularization options (e.g., `none`/`orth1.0`).
     * To generate all experiments for a single dataset, you can run:```make build-all path=experiments/configs/Exchange/```
     * Likewise, to generate all experiments for all datasets, you can run:```make build-all path=experiments/configs/*```

3. Run all experiments: `sh run.sh`

4. Finally, you can observe the results on tensorboard
`tensorboard --logdir storage/experiments/` or view the storage/experiments/**/metrics.npy file. The hyperparameters were chosen based on the validation MSE.


## Acknowledgements

The implementation of DeepRRTime heavily relies on the original DeepTime implementation (https://github.com/salesforce/DeepTime). We thank the original authors for open-sourcing their work. Compared to the original implementation, only the following python files were updated:
* experiments/base.py
* experiments/forecast.py
* models/DeepTIMe.py
* models/modules/regressors.py


## Citation

To cite our work please use the following reference:
<pre>
@article{
    sastry2025deeprrtime,
    title={Deep{RRT}ime: Robust Time-series Forecasting with a Regularized {INR} Basis},
    author={Chandramouli Shama Sastry and Mahdi Gilany and Kry Yik-Chau Lui and Martin Magill and Alexander Pashevich},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=uDRzORdPT7},
    note={}
}</pre>
