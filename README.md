# Transformers in Time Series  

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/qingsongedu/time-series-transformers-review)
[![Visits Badge](https://badges.pufler.dev/visits/qingsongedu/time-series-transformers-review)](https://badges.pufler.dev/visits/qingsongedu/time-series-transformers-review)
<!-- ![Forks](https://img.shields.io/github/forks/qingsongedu/time-series-transformers-review) -->


A professionally curated list of awesome resources (paper, code, data, etc.) on **Transformers in Time Series**, which is first work to comprehensively and systematically summarize the recent advances of Transformers for modeling time series data to the best of our knowledge.

We will continue to update this list with newest resources. If you found any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

## Survey paper

[**Transformers in Time Series: A Survey**](https://arxiv.org/pdf/2202.07125.pdf)

[Qingsong Wen](https://sites.google.com/site/qingsongwen8/), Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, [Junchi Yan](https://thinklab.sjtu.edu.cn/) and [Liang Sun](https://scholar.google.com/citations?user=8JbrsgUAAAAJ&hl=en).

#### If you find this repository helpful for your work, please kindly cite our survey paper.

```bibtex
@article{wen2022tstransformers,
  title={Transformers in Time Series: A Survey},
  author={Wen, Qingsong and Zhou, Tian and Zhang, Chaoli and Chen, Weiqi and Ma, Ziqing and Yan, Junchi and Sun, Liang},
  journal={arXiv preprint arXiv:2202.07125},
  year={2022}
}
```

## Taxonomy of Transformers for time series modeling
<!-- ![xxx](TS_Xformer_V2.jpg) -->
<img src="TS_Xformer_V2.jpg" width=700 align=middle> <br />


## Application Domains of Time Series Transformers

### Transformers in Forecasting
#### Time Series Forecasting
* Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting, in *NeurIPS* 2019. [\[paper\]](https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html) [\[code\]](https://github.com/mlpotter/Transformer_Time_Series)
* Informer: Beyond efficient transformer for long sequence time-series forecasting, in *AAAI* 2021. [\[paper\]](https://www.aaai.org/AAAI21Papers/AAAI-7346.ZhouHaoyi.pdf) [\[official code\]](https://github.com/zhouhaoyi/Informer2020) [\[dataset\]](https://github.com/zhouhaoyi/ETDataset) 
* Adversarial sparse transformer for time series forecasting, in *NeurIPS* 2020. [\[paper\]](https://proceedings.neurips.cc/paper/2020/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html) [\[code\]](https://github.com/hihihihiwsf/AST)
* Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting, in *NeurIPS* 2021. [\[paper\]](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) [\[official code\]](https://github.com/thuml/autoformer)
* Probabilistic Transformer For Time Series Analysis, in *NeurIPS* 2021. [\[paper\]](https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html)  
* Temporal fusion transformers for interpretable multi-horizon time series forecasting, in *International Journal of Forecasting* 2021. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S0169207021000637) [\[code\]](https://github.com/mattsherar/Temporal_Fusion_Transform)
* SSDNet: State Space Decomposition Neural Network for Time Series Forecasting, in *ICDM* 2021, [\[paper\]](https://arxiv.org/abs/2112.10251)
* From Known to Unknown: Knowledge-guided Transformer for Time-Series Sales Forecasting in Alibaba, in *arXiv* 2021. [\[paper\]](https://arxiv.org/abs/2109.08381)
* Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting, in *ICLR* 2022. [\[paper\]](https://openreview.net/forum?id=0EXmFzUn5I)
* FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting, in *ICML* 2022. [\[paper\]](https://arxiv.org/abs/2201.12740)
* TCCT: Tightly-coupled convolutional transformer on time series forecasting, in *Neurocomputing* 2022. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S0925231222000571)


 #### Spatio-Temporal Forecasting
* Spatio-temporal graph transformer networks for pedestrian trajectory prediction, in *ECCV* 2020. [\[paper\]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1636_ECCV_2020_paper.php) [\[official code\]](https://github.com/Majiker/STAR)
* Spatial-temporal transformer networks for traffic flow forecasting, in *arXiv* 2020. [\[paper\]](https://arxiv.org/abs/2001.02908) [\[official code\]](https://github.com/xumingxingsjtu/STTN)
* Traffic transformer: Capturing the continuity and periodicity of time series for traffic forecasting, in *Transactions in GIS* 2022. [\[paper\]](https://coolgiserz.github.io/publication/traffic-transformer-capturing-the-continuity-and-periodicity-of-time-series-for-traffic-forecasting/traffic-transformer-capturing-the-continuity-and-periodicity-of-time-series-for-traffic-forecasting.pdf)

 #### Event Forecasting
* Self-attentive Hawkes process, in *ICML* 2020. [\[paper\]](http://proceedings.mlr.press/v119/zhang20q.html) [\[official code\]](https://github.com/QiangAIResearcher/sahp_repo)
* Transformer Hawkes process, in *ICML* 2020. [\[paper\]](https://proceedings.mlr.press/v119/zuo20a.html) [\[official code\]](https://github.com/SimiaoZuo/Transformer-Hawkes-Process)
* Transformer Embeddings of Irregularly Spaced Events and Their Participants, in *ICLR* 2022. [\[paper\]](https://openreview.net/forum?id=Rty5g9imm7H) [\[official code\]](https://github.com/yangalan123/anhp-andtt)


### Transformers in Anomaly Detection
* Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, in *ICLR* 2022. [\[paper\]](https://arxiv.org/abs/2110.02642) [\[code\]](https://github.com/spencerbraun/anomaly_transformer_pytorch)
* TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data, in *VLDB* 2022. [\[paper\]](https://arxiv.org/abs/2201.07284) [\[official code\]](https://github.com/imperial-qore/tranad)
* Learning graph structures with transformer for multivariate time series anomaly detection in IoT, in *IEEE Internet of Things Journal* 2021. [\[paper\]](https://arxiv.org/abs/2104.03466) [\[official code\]](https://github.com/ZEKAICHEN/GTA)
* Spacecraft Anomaly Detection via Transformer Reconstruction Error, in *ICASSE* 2019. [\[paper\]](http://www.utias.utoronto.ca/wp-content/uploads/2019/07/88-Spacecraft-anomaly-detection-via-transformer-reconstruction-error.pdf)
* Unsupervised Anomaly Detection in Multivariate Time Series through Transformer-based Variational Autoencoder, in *CCDC* 2021. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/9601669)
* Variational Transformer-based anomaly detection approach for multivariate time series, in *Measurement* 2022. [\[paper\]](https://www.sciencedirect.com/science/article/abs/pii/S0263224122000914)

### Transformers in Classification
* A transformer-based framework for multivariate time series representation learning, in *KDD* 2021. [\[paper\]](https://arxiv.org/abs/2010.02803) [\[official code\]](https://github.com/gzerveas/mvts_transformer)
* Voice2series: Reprogramming acoustic models for time series classification, in *ICML* 2021. [\[paper\]](https://arxiv.org/abs/2106.09296) [\[official code\]](https://github.com/huckiyang/Voice2Series-Reprogramming)
* Gated Transformer Networks for Multivariate Time Series Classification, in *arXiv* 2021. [\[paper\]](https://arxiv.org/abs/2103.14438) [\[official code\]](https://github.com/ZZUFaceBookDL/GTN)
* Self-attention for raw optical satellite time series classification, in *ISPRS Journal of Photogrammetry and Remote Sensing* 2020. [\[paper\]](https://arxiv.org/abs/1910.10536) [\[official code\]](https://github.com/marccoru/crop-type-mapping)
* Self-supervised pretraining of transformers for satellite image time series classification, in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing* 2020. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/9252123)
* Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series, in *ACM TKDD* 2022. [\[paper\]](https://people.cs.vt.edu/~reddy/papers/TKDD22a.pdf)

## Time Series Related Survey
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2002.12478)
* Neural temporal point processes: a review, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2104.03528v5)
* Time-series forecasting with deep learning: a survey, in *Philosophical Transactions of the Royal Society A* 2021. [\[paper\]](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0209)
* Deep learning for time series forecasting: a survey, in *Big Data* 2021. [\[paper\]](https://www.liebertpub.com/doi/abs/10.1089/big.2020.0159)
* Neural forecasting: Introduction and literature overview, in *arXiv* 2020. [\[paper\]](https://arxiv.org/abs/2004.10240) 
* Deep learning for anomaly detection in time-series data: review, analysis, and guidelines, in *Access* 2021. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/9523565) 
* A review on outlier/anomaly detection in time series data, in *ACM Computing Surveys* 2021. [\[paper\]](https://arxiv.org/abs/2002.04236)
* A unifying review of deep and shallow anomaly detection, in *Proceedings of the IEEE* 2021. [\[paper\]](http://128.84.4.34/abs/2009.11732)
* Deep learning for time series classification: a review, in *Data Mining and Knowledge Discovery* 2019. [\[paper\]](https://link.springer.com/article/10.1007/s10618-019-00619-1?sap-outbound-id=11FC28E054C1A9EB6F54F987D4B526A6EE3495FD&mkt-key=005056A5C6311EE999A3A1E864CDA986)

 
## Transformer/Attention Survey in Other Disciplines
* A survey on visual transformer, in *IEEE TPAMI* 2022. [\[paper\]](https://arxiv.org/abs/2012.12556)
* A General Survey on Attention Mechanisms in Deep Learning, in *IEEE TKDE* 2022. [\[paper\]](https://personal.eur.nl/frasincar/papers/TKDE2022/tkde2022.pdf)
* Attention, please! A survey of neural attention models in deep learning, in *Artificial Intelligence Review* 2022. [\[paper\]](https://link.springer.com/article/10.1007/s10462-022-10148-x)
* Attention mechanisms in computer vision: A survey, in *Computational Visual Media* 2022. [\[paper\]](https://link.springer.com/article/10.1007/s41095-022-0271-y)
* Survey: Transformer based video-language pre-training, in _AI Open_ 2022. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S2666651022000018)
* Transformers in vision: A survey, in *ACM Computing Surveys* 2021. [\[paper\]](https://arxiv.org/abs/2101.01169)
* Pre-trained models: Past, present and future, in *AI Open* 2021. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S2666651021000231)
* An attentive survey of attention models, in *ACM TIST* 2021. [\[paper\]](https://arxiv.org/abs/1904.02874)
* Attention in natural language processing, in *IEEE TNNLS* 2020. [\[paper\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9194070)
* Pre-trained models for natural language processing: A survey, in *Science China Technological Sciences* 2020. [\[paper\]](https://link.springer.com/article/10.1007/s11431-020-1647-3)
* Efficient transformers: A survey, in *arXiv* 2020. [\[paper\]](https://arxiv.org/abs/2009.06732)
* A Survey of Transformers, in _arXiv_ 2021. [\[paper\]](https://arxiv.org/abs/2106.04554)
* A Survey of Vision-Language Pre-Trained Models, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2202.10936)
* Video Transformers: A Survey, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2201.05991)
* Transformer for Graphs: An Overview from Architecture Perspective, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2202.08455)
* Transformers in Medical Imaging: A Survey, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2201.09873) 
* A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2201.05337) 

