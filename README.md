# One AI Model for Multi-scenario Reconstructions: Physics-informed Synthetic Data Boosts Generalizable Deep Learning for Fast MRI
This work presents a Physics-Informed Synthetic data learning Framework for fast MRI, called PISF, which is the first to enable generalizable DL for multi-scenario MRI reconstruction using solely one trained model.

1) We demonstrate that training DL models on synthetic data, integrated with enhanced learning techniques, can achieve comparable or even better *in vivo* MRI reconstruction compared to models trained on a matched realistic dataset—PISF reduces the demand for real-world MRI data by up to 96%. 
2) Our PISF shows impressive generalizability in multi-vendor multi-center imaging—it can reconstruct high-quality images of 4 anatomies and 5 contrasts across 5 vendors and centers using a single trained network.
3) PISF’s superior adaptability to patients has been verified through 10 experienced doctors’ evaluations (4 neuro radiologists and 1 neurosurgeon for brain tumor patients, and 3 cardiac radiologists and 2 cardiologists for myocardial hypertrophy patients)—its overall image quality steps into the excellent level in reader study.

In summary, PISF provides a feasible and cost-effective way to markedly boost the widespread usage of DL in various fast MRI applications, while freeing from the intractable ethical and practical considerations of *in vivo* human data acquisitions. 
![OverallConcept_PISF](https://github.com/wangziblake/PISF/blob/main/Figure/OverallConcept_PISF.png)

The preprint paper can be seen at https://doi.org/10.48550/arXiv.2307.13220.

This paper has been accepted by Medical Image Analysis (2025) at https://doi.org/10.1016/j.media.2025.103616.

Email: Xiaobo Qu (quxiaobo@xmu.edu.cn) CC: Zi Wang (zi.wang123@imperial.ac.uk)

Homepage: http://csrc.xmu.edu.cn


## Testing codes of PISF
The testing codes of PISF are released here.

Install conda environment:
```bash
conda env create -f environment
```
Run the main code for reconstruction:
```python
python PISF_Recon_Enhance.py
```

Python environment should be: python=3.6.13, pytorch=1.10.1

Implementation tips: If you want to test on your own collected data, they should be stored in the same format as the demo data we provided and be 8-coil or compressed/extended to 8-coil. (Note: The code of a much more coil-number-flexible version PISFC proposed in the published paper is currently under collation.)

Data availability: All used public datasets are available at their websites, including https://fastmri.org, http://www.mridata.org, and https://ocmr.info. Other in-house MRI datasets from our own collection are available from the corresponding author upon reasonable request.

Note: The software is used for testing only, and cannot be used commercially.


## Citation
If you want to use the code, please cite the following paper:

Zi Wang et al., One for Multiple: Physics-informed Synthetic Data Boosts Generalizable Deep Learning for Fast MRI Reconstruction, Medical Image Analysis, 103: 103616, 2025.
