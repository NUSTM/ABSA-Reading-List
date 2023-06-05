# Aspect-Based Sentiment Analysis Reading List
Reading List of aspect-based sentiment analysis (ABSA), Cross-Domain ABSA, and Multi-Modal ABSA, maintained by Rui Xia, Jianfei Yu, Hongjie Cai, Zengzhi Wang, Junjie Li, and Yan Ling from Text Mining Group of Nanjing University of Science & Technology (NUSTM).

<!-- This directory contains the aspect-based sentiment analysis reading list .
maintained by Rui Xia, Jianfei Yu, and Hongjie Cai from Text Mining Group of Nanjing University of Science & Technology (NUSTM) -->


### Contents


- [1. ABSA](#1-absa)
  - [1.1 Aspect-Oriented Sentiment Classification](#11-aspect-oriented-sentiment-classification)
  - [1.2 Aspect Extraction](#12-aspect-extraction)
  - [1.3 Opinion Extraction](#13-opinion-extraction)
  - [1.4 Category Detection](#14-category-detection)
  - [1.5 Aspect-Opinion Co-Extraction](#15-aspect-opinion-co-extraction)
  - [1.6 Aspect-Oriented Opinion Extraction](#16-aspect-oriented-opinion-extraction)
  - [1.7 Aspect-Opinion Pair Extraction](#17-aspect-opinion-pair-extraction)
  - [1.8 Aspect-Sentiment Pair Extraction](#18-aspect-sentiment-pair-extraction)
  - [1.9 Category-Oriented Sentiment Classification](#19-category-oriented-sentiment-classification)
  - [1.10 Category-Sentiment Hierarchical Classification](#110-category-sentiment-hierarchical-classification)
  - [1.11 Aspect-Category-Sentiment Triple Extraction](#111-aspect-category-sentiment-triple-extraction)
  - [1.12 Aspect-Opinion-Sentiment Triple Extraction](#112-aspect-opinion-sentiment-triple-extraction)
  - [1.13 Aspect-Category-Opinion-Sentiment Quadruple Extraction](#113-aspect-category-opinion-sentiment-quadruple-extraction)
- [2. Cross-Domain ABSA](#2-cross-domain-absa)
  - [2.1 Cross-Domain Aspect Extraction](#21-cross-domain-aspect-extraction)
  - [2.2 Cross-Domain Aspect-Opinion Co-Extraction](#22-cross-domain-aspect-opinion-co-extraction)
  - [2.3 Cross-Domain Aspect-Oriented Sentiment Classification](#23-cross-domain-aspect-oriented-sentiment-classification)
  - [2.4 Cross-Domain Aspect-Sentiment Pair Extraction](#24-cross-domain-aspect-sentiment-pair-extraction)
- [3. Multi-Modal ABSA](#3-multi-modal-absa)
  - [3.1 Multi-Modal Aspect Extraction (& Multi-Modal Named Entity Recognition)](#31-multi-modal-aspect-extraction--multi-modal-named-entity-recognition)
  - [3.2 Multi-Modal Category-Oriented Sentiment Classification](#32-multi-modal-category-oriented-sentiment-classification)
  - [3.3 Multi-Modal Aspect-Oriented Sentiment Classification](#33-multi-modal-aspect-oriented-sentiment-classification)
  - [3.4 Multi-Modal Aspect-Sentiment Pair Extraction](#34-multi-modal-aspect-sentiment-pair-extraction)

 ## 1. ABSA
 ### 1.1 Aspect-Oriented Sentiment Classification

1. Jiahao Cao, Rui Liu, Huailiang Peng, Lei Jiang, Xu Bai. **Aspect Is Not You Need: No-aspect Differential Sentiment Framework for Aspect-based Sentiment Analysis**. NAACL 2022. [[paper]](https://aclanthology.org/2022.naacl-main.115.pdf)

1. Zheng Zhang, Zili Zhou, Yanna Wang. **SSEGCN: Syntactic and Semantic Enhanced Graph Convolutional Network for Aspect-based Sentiment Analysis**. NAACL 2022. [[paper]](https://aclanthology.org/2022.naacl-main.362.pdf) [[code]](https://github.com/zhangzheng1997/ssegcn-absa)

1. Ehsan Hosseini-Asl, Wenhao Liu, Caiming Xiong. **A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis**. NAACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-naacl.58.pdf) [[code]](https://github.com/salesforce/fewshot_absa)

1. Chenhua Chen, Zhiyang Teng, Zhongqing Wang and Yue Zhang. **Discrete Opinion Tree Induction for Aspect-based Sentiment Analysis**. ACL 2022. [[paper]](https://aclanthology.org/2022.acl-long.145.pdf) [[code]](https://aclanthology.org/attachments/2022.acl-long.145.software.zip)

1. Yiming Zhang, Min Zhang, Sai Wu, Junbo Zhao (Jake) . **Towards Unifying the Label Space for Aspect- and Sentence-based Sentiment Analysis**. ACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-acl.3.pdf)[[code]](https://github.com/yiming-zh/DPL)

1. Shuo Liang, Wei Wei, , Xian-Ling Mao, Fei Wang, Zhiyong He. **BiSyn-GAT+: Bi-Syntax Aware Graph Attention Network for Aspect-based Sentiment Analysis**. ACL  Findings 2022. [[paper]](https://aclanthology.org/2022.findings-acl.144.pdf)[[code]](https://github.com/CCIIPLab/BiSyn_GAT_plus)

1. Kai Zhang, Kun Zhang, Mengdi Zhang, Hongke Zhao, Qi Liu, Wei Wu, Enhong Chen. **Incorporating Dynamic Semantics into Pre-Trained Language Model for Aspect-based Sentiment Analysis**. ACL  Findings 2022. [[paper]](https://aclanthology.org/2022.findings-acl.285.pdf)

1. Bo Wang, Tao Shen, Guodong Long, Tianyi Zhou, Yi Chang. **Eliminating Sentiment Bias for Aspect-Level Sentiment Classification with Unsupervised Opinion Extraction**. EMNLP Findings 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.258/) [[code]](https://github.com/wangbo9719/SARL_ABSA)

1. Zeguan Xiao, Jiarun Wu, Qingliang Chen, Congjian Deng. **BERT4GCN: Using BERT Intermediate Layers to Augment GCN for Aspect-based Sentiment Classification**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.724/)

1. Zixuan Ke, Bing Liu, Hu Xu, Lei Shu. **CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.550/) [[code]](https://github.com/ZixuanKe/PyContinual)

1. Ronald Seoh, Ian Birle, Mrinal Tak, Haw-Shiuan Chang, Brian Pinette, Alfred Hough. **Open Aspect Target Sentiment Classification with Natural Language Prompts**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.509/) [[code]](https://github.com/ronaldseoh/atsc_prompts/)

1. Ting-Wei Hsu, Chung-Chi Chen, Hen-Hsen Huang, Hsin-Hsi Chen. **Semantics-Preserved Data Augmentation for Aspect-Based Sentiment Analysis**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.362/) [[code]](https://github.com/Quant-NLP/SPDAug-ABSA)

1. Han Qin, Guimin Chen, Yuanhe Tian, Yan Song. **Improving Federated Learning for Aspect-based Sentiment Analysis via Topic Memories**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.321/) [[code]](https://github.com/cuhksz-nlp/ASA-TM)

1. Yuxiang Zhou, Lejian Liao, Yang Gao, Zhanming Jie, Wei Lu. **To be Closer: Learning to Link up Aspects with Opinions**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.317/) [[code]](https://github.com/zyxnlp/ACLT)

1. Zhengyan Li, Yicheng Zou, Chong Zhang, Qi Zhang, Zhongyu Wei. **Learning Implicit Sentiment in Aspect-based Sentiment Analysis with Supervised Contrastive Pre-Training**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.22/) [[code]](https://github.com/Tribleave/SCAPT-ABSA)

1. Ruifan Li, Hao Chen, Fangxiang Feng, Zhanyu Ma, Xiaojie Wang, Eduard Hovy. **Dual Graph Convolutional Networks for Aspect-based Sentiment Analysis**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.494.pdf)

1. Yuanhe Tian, Guimin Chen, Yan Song. **Aspect-based Sentiment Analysis with Type-aware Graph Convolutional Networks and Layer Ensemble**. NAACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.231.pdf)

1. Junqi Dai, Hang Yan, Tianxiang Sun, Pengfei Liu, Xipeng Qiu. **Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa**. NAACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.146.pdf)

1. Xiaochen Hou, Peng Qi, Guangtao Wang, Rex Ying, Jing Huang, Xiaodong He, Bowen Zhou. **Graph Ensemble Learning over Multiple Dependency Trees for Aspect-level Sentiment Classification**. NAACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.229.pdf)

1. Zixuan Ke, Hu Xu, Bing Liu. **Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks**. NAACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.378.pdf)

1. Andrew Moore, Jeremy Barnes. **Multi-task Learning of Negation and Speculation for Targeted Sentiment Classification**. NAACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.227.pdf)

1. Zhengxuan Wu, Desmond C. Ong. **Context-Guided BERT for Targeted Aspect-Based Sentiment Analysis**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17659/17466)

1. Rohan Kumar Yadav, Lei Jiao, Ole-Christoffer Granmo, Morten Goodwin. **Human-Level Interpretable Learning for Aspect-Based Sentiment Analysis**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17671/17478)

1. Xiaoyu Xing, Zhijing Jin, Di Jin, Bingning Wang, Qi Zhang, Xuanjing Huang. **Tasty Burgers, Soggy Fries: Probing Aspect Robustness in Aspect-Based Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.292.pdf)

1. Lu Xu, Lidong Bing, Wei Lu, Fei Huang. **Aspect Sentiment Classification with Aspect-Specific Opinion Spans**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.288.pdf)

1. Chenhua Chen, Zhiyang Teng, Yue Zhang. **Inducing Target-Specific Latent Structures for Aspect Sentiment Classification**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.451.pdf)

1. Mi Zhang, Tieyun Qian. **Convolution over Hierarchical Syntactic and Lexical Graphs for Aspect Level Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.286.pdf)

1. Zehui Dai, Cheng Peng, Huajie Chen, Yadong Ding. **A Multi-Task Incremental Learning Framework with Category Name Embedding for Aspect-Category Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.565.pdf)

1. Yuncong Li, Cunxiang Yin, Sheng-hua Zhong, Xu Pan. **Multi-Instance Multi-Label Learning Networks for Aspect-Category Sentiment  Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.287.pdf)

1. Hao Tang, Donghong Ji, Chenliang Li, Qiji Zhou. **Dependency Graph Enhanced Dual-transformer Structure for Aspect-based Sentiment Classification**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.588.pdf)

1. Xiao Chen, Changlong Sun, Jingjing Wang, Shoushan Li, Luo Si, Min Zhang, Guodong Zhou. **Aspect Sentiment Classification with Document-level Sentiment Preference Modeling**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.338.pdf)

1. Kai Wang, Weizhou Shen, Yunyi Yang, Xiaojun Quan, Rui Wang. **Relational Graph Attention Network for Aspect-based Sentiment Analysis**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.295.pdf) [[code]](https://github.com/shenwzh3/RGAT-ABSA)

1. Minh Hieu Phan, Philip O. Ogunbona. **Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.293.pdf)

1. Jingjing Wang, Changlong Sun, Shoushan Li, Xiaozhong Liu, Min Zhang, Luo Si and Guodong Zhou. **Aspect Sentiment Classification Towards Question-Answering with Reinforced Bidirectional Attention Network**. ACL 2019. [[paper]](https://www.aclweb.org/anthology/P19-1345.pdf)

1. Hu Xu, Bing Liu, Lei Shu and Philip S. Yu. **BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis**. NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1242.pdf) [[code]](https://github.com/howardhsu/BERT-for-RRC-ABSA)

1. Binxuan Huang and Kathleen Carley. **Syntax-Aware Aspect Level Sentiment Classification with Graph Attention Networks**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1549.pdf)

1. Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao and Min Yang. **A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1654.pdf)

1. Chunning Du, Haifeng Sun, Jingyu Wang, Qi Qi, Jianxin Liao, Tong Xu and Ming Liu. **Capsule Network with Interactive Attention for Aspect-Level Sentiment Classification**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1551.pdf)

1. Mengting Hu, Shiwan Zhao, Li Zhang, Keke Cai, Zhong Su, Renhong Cheng and Xiaowei Shen. **CAN: Constrained Attention Networks for Multi-Aspect Sentiment Analysis**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1467.pdf)

1. Kai Sun, Richong Zhang, Samuel Mensah, Yongyi Mao and Xudong Liu. **Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1569.pdf)

1. Chen Zhang, Qiuchi Li and Dawei Song. **Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1464.pdf)

1. Yunlong Liang, Fandong Meng, Jinchao Zhang, Jinan Xu, Yufeng Chen and Jie Zhou. **A Novel Aspect-Guided Deep Transition Model for Aspect Based Sentiment Analysis**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1559.pdf)

1. Shuai Wang , Sahisnu Mazumder , Bing Liu , Mianwei Zhou and Yi Chang. **Target-Sensitive Memory Networks for Aspect Sentiment Classification**. ACL 2018. [[paper]](https://www.aclweb.org/anthology/P18-1088.pdf)

1. Wei Xue and Tao Li. **Aspect Based Sentiment Analysis with Gated Convolutional Networks**. ACL 2018. [[paper]](https://www.aclweb.org/anthology/P18-1234.pdf)

1. Ruidan He, Wee Sun Lee, Hwee Tou Ng and Daniel Dahlmeier. **Exploiting Document Knowledge for aspect-level sentiment classification**. ACL 2018. [[paper]](https://www.aclweb.org/anthology/P18-2092.pdf)

1. Navonil Majumder, Soujanya Poria, Alexander Gelbukh, Md. Shad Akhtar, Erik Cambria and Asif Ekba. **Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis**. EMNLP 2018. [[paper]](https://www.aclweb.org/anthology/D18-1377.pdf)

1. Jingjing Wang , Jie Li , Shoushan Li, Yangyang Kang , Min Zhang , Luo Si and Guodong Zhou. **Aspect Sentiment Classification with both Word-level and Clause-level Attention Networks**. IJCAI 2018. [[paper]](https://www.ijcai.org/Proceedings/2018/0617.pdf)

1. Jun Yang, Runqi Yang, Chongjun Wang and Junyuan Xie. **Multi-Entity Aspect-Based Sentiment Analysis with Context, Entity and Aspect Memory**. AAAI 2018. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17036/16171)

1. Yi Tay, Luu Anh Tuan and Siu Cheung Hui. **Learning to Attend via Word-Aspect Associative Fusion for Aspect-Based Sentiment Analysis**. AAAI 2018. [[paper]](https://arxiv.org/abs/1712.05403)

1. Bailin Wang and Wei Lu. **Learning Latent Opinions for Aspect-Level Sentiment Classification**. AAAI 2018. [[paper]](http://www.statnlp.org/wp-content/uploads/papers/2018/Learning-Latent/absa.pdf)

1. Xin Li, Lidong Bing, Wai Lam and Bei Sh. **Transformation Networks for Target-Oriented Sentiment Classification**. ACL 2018. [[paper]](https://www.aclweb.org/anthology/P18-1087.pdf)

1. Binxuan Huang, Kathleen Carley. **Parameterized Convolutional Neural Networks for Aspect Level Sentiment Classification**. EMNLP 2018. [[paper]](https://aclanthology.org/D18-1136/)

1. Shiliang Zheng and Rui Xia. **Left-Center-Right Separated Neural Network for Aspect-based Sentiment Analysis with Rotatory Attention**. Arxiv 2018. [[paper]](https://arxiv.org/abs/1802.00892)

1. Peng Chen, Zhongqian Sun, Lidong Bing and Wei Yang. **Recurrent Attention Network on Memory for Aspect Sentiment Analysis**. EMNLP 2017. [[paper]](https://www.aclweb.org/anthology/D17-1047.pdf)

1. Dehong Ma, Sujian Li, Xiaodong Zhang and Houfeng Wang. **Interactive Attention Networks for Aspect-Level Sentiment Classification**. IJCAI 2017. [[paper]](https://www.ijcai.org/Proceedings/2017/0568.pdf)

1. Meishan Zhang, Yue Zhang and Duy-Tin Vo. **Gated Neural Networks for Targeted Sentiment Analysis**. AAAI 2016. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12074/12065)

1. Duyu Tang, Bing Qin and Ting Liu. **Aspect Level Sentiment Classification with Deep Memory Network**. EMNLP 2016. [[paper]](https://www.aclweb.org/anthology/D16-1021.pdf)

1. Yequan Wang, Minlie Huang, Li Zhao and Xiaoyan Zhu. **Attention-based LSTM for Aspect-level Sentiment Classification**. EMNLP 2016. [[paper]](https://www.aclweb.org/anthology/D16-1058.pdf)

1. Duyu Tang, Bing Qin, Xiaocheng Feng and Ting Liu. **Effective LSTMs for Target-Dependent Sentiment Classification with Long Short Term Memory**. COLING 2016. [[paper]](https://www.aclweb.org/anthology/C16-1311.pdf)

1. Li Dong, Furu Wei, Chuanqi Tan, Duyu Tang, Ming Zhou, Ke Xu. **Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification**. ACL 2014. [[paper]](https://aclanthology.org/P14-2009/)[[dataset]](https://docs.google.com/forms/d/e/1FAIpQLSd6HhEitpzv-vpdrB63Jbx-fYWohnOZRCUb0ibKjpO21Q_tIQ/viewform)

1. Long Jiang, Mo Yu, Ming Zhou, Xiaohua Liu, Tiejun Zhao. **Target-dependent Twitter Sentiment Classification**. ACL 2011. [[paper]](https://aclanthology.org/P11-1016.pdf)

### 1.2 Aspect Extraction

1. Ehsan Hosseini-Asl, Wenhao Liu, Caiming Xiong. **A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis**. NAACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-naacl.58.pdf) [[code]](https://github.com/salesforce/fewshot_absa)

1. Chang-You Tai, Ming-Yao Li, Lun-Wei Ku. **Hyperbolic Disentangled Representation for Fine-Grained Aspect Extraction**. AAAI 2022. [[paper]](https://www.aaai.org/AAAI22Papers/AAAI-6574.TaiCY.pdf) [[code]](https://github.com/johnnyjana730/HDAE/)

1. Qianlong Wang, Zhiyuan Wen, Qin Zhao, Min Yang, Ruifeng Xu. **Progressive Self-Training with Discriminator for Aspect Term Extraction**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.23/) [[code]](https://github.com/qlwang25/progressive_self_training)

1. Zhuang Chen, Tieyun Qian. **Bridge-Based Active Domain Adaptation for Aspect Term Extraction**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.27.pdf)

1. Tian Shi, Liuqing Li, Ping Wang, Chandan K. Reddy. **A Simple and Effective Self-Supervised Contrastive Learning Framework for Aspect Detection**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17628/17435)

1. Kun Li, Chengbo Chen, Xiaojun Quan, Qing Ling, Yan Song. **Conditional Augmentation for Aspect Term Extraction via Masked Sequence-to-Sequence Generation**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.631.pdf)

1. Stéphan Tulkens, Andreas van Cranenburgh. **Embarrassingly Simple Unsupervised Aspect Extraction**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.290.pdf)

1. Zhenkai Wei, Yu Hong, Bowei Zou, Meng Cheng, Jianmin YAO. **Don’t Eclipse Your Arts Due to Small Discrepancies: Boundary Repositioning with a Pointer Network for Aspect Extraction**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.339.pdf) [[code]](https://www.aclweb.org/anthology/attachments/2020.acl-main.339.Software.zip)

1. Zhuang Chen, Tieyun Qian. **Enhancing Aspect Term Extraction with Soft Prototypes**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.164.pdf)

1. Dehong Ma, Sujian Li, Fangzhao Wu, Xing Xie, Houfeng Wang. **Exploring Sequence-to-Sequence Learning in Aspect Term Extraction**. ACL 2019. [[paper]](https://aclanthology.org/P19-1344.pdf)

1. Hongliang Dai, Yangqiu Song. **Neural Aspect and Opinion Term Extraction with Mined Rules as Weak Supervision**. ACL 2019. [[paper]](https://aclanthology.org/P19-1520.pdf)

1. Ming Liao, Jing Li, Haisong Zhang, Lingzhi Wang, Xixin Wu, Kam-Fai Wong. **Coupling Global and Local Context for Unsupervised Aspect Extraction**. EMNLP 2019. [[paper]](https://aclanthology.org/D19-1465.pdf)

1. Xin Li, Lidong Bing, Piji Li, Wai Lam, Zhimou Yang. **Aspect Term Extraction with History Attention and Selective Transformation**. IJCAI 2018. [[paper]](https://www.ijcai.org/Proceedings/2018/0583.pdf)

1. Hu Xu, Bing Liu, Lei Shu, Philip S. Yu. **Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction**. ACL 2018. [[paper]](https://aclanthology.org/P18-2094.pdf)

1. Ruidan He, Wee Sun Lee, Hwee Tou Ng, Daniel Dahlmeier. **An Unsupervised Neural Attention Model for Aspect Extraction**. ACL 2017. [[paper]](https://aclanthology.org/P17-1036.pdf)

1. Xin Li, Wai Lam. **Deep Multi-Task Learning for Aspect Term Extraction with Memory Interaction**. EMNLP 2017. [[paper]](https://aclanthology.org/D17-1310.pdf)

1. Yichun Yin, Furu Wei, Li Dong, Kaimeng Xu, Ming Zhang, Ming Zhou. **Unsupervised Word and Dependency Path Embeddings for Aspect Term Extraction**. IJCAI 2016. [[paper]](https://www.ijcai.org/Proceedings/16/Papers/423.pdf)

1. Fangtao Li, Chao Han, Minlie Huang, Xiaoyan Zhu, Ying-Ju Xia, Shu Zhang, Hao Yu. **Structure-Aware Review Mining and Summarization**. COLING 2010. [[paper]](https://aclanthology.org/C10-1074.pdf)

1. Wei Jin, Hung Hay Ho. **A Novel Lexicalized HMM-based Learning Framework for Web Opinion Mining**. ICML 2009. [[paper]](http://people.cs.pitt.edu/~huynv/research/aspect-sentiment/A%20novel%20lexicalized%20HMM-based%20learning%20framework%20for%20web%20opinion%20mining.pdf)


### 1.3 Opinion Extraction


### 1.4 Category Detection

1. Ehsan Hosseini-Asl, Wenhao Liu, Caiming Xiong. **A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis**. NAACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-naacl.58.pdf) [[code]](https://github.com/salesforce/fewshot_absa)

1. Thi-Nhung Nguyen, Kiem-Hieu Nguyen, Young-In Song, Tuan-Dung Cao. **An Uncertainty-Aware Encoder for Aspect Detection**. EMNLP Findings 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.69/) 
 
1. Jian Liu, Zhiyang Teng, Leyang Cui, Hanmeng Liu, Yue Zhang. **Solving Aspect Category Sentiment Analysis as a Text Generation Task**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.361/) [[code]](https://github.com/lgw863/ACSA-generation)

1. Mengting Hu, Shiwan Zhao, Honglei Guo, Chao Xue, Hang Gao, Tiegang Gao, Renhong Cheng, Zhong Su. **Multi-Label Few-Shot Learning for Aspect Category Detection**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.495.pdf)


### 1.5 Aspect-Opinion Co-Extraction

1. Meixi Wu, Wenya Wang, Sinno Jialin Pan. **Deep Weighted MaxSAT for Aspect-based Opinion Extraction**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.453.pdf)

1. Jianfei Yu, Jing Jiang, Rui Xia. **Global inference for aspect and opinion terms co-extraction based on multi-task neural networks**. IEEE TASLP 2018. [[paper]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5163&context=sis_research)

1. Wenya Wang, Sinno Jialin Pan, Daniel Dahlmeier, Xiaokui Xiao. **Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms**. AAAI 2017. [[paper]](https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf)

1. Wenya Wang, Sinno Jialin Pan, Daniel Dahlmeier, Xiaokui Xiao. **Recursive Neural Conditional Random Fields for Aspect-based Sentiment Analysis**. EMNLP 2016. [[paper]](https://www.aclweb.org/anthology/D16-1059.pdf)

1. Pengfei Liu, Shafiq Joty, Helen Meng. **Fine-grained Opinion Mining with Recurrent Neural Networks and Word Embeddings**. EMNLP 2015. [[paper]](https://aclanthology.org/D15-1168.pdf)

### 1.6 Aspect-Oriented Opinion Extraction

1. Junjie Li, Jianfei Yu, and Rui Xia. **Generative Cross-Domain Data Augmentation for Aspect and Opinion Co-Extraction**. NAACL 2022. [[paper]](https://aclanthology.org/2022.naacl-main.312.pdf) [[code]](https://github.com/nustm/gcdda)

1. Yue Mao, Yi Shen, Jingchao Yang, Xiaoying Zhu, Longjun Cai. **Seq2Path: Generating Sentiment Tuples as Paths of a Tree**. ACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-acl.174.pdf)[[code]](https://aclanthology.org/attachments/2022.findings-acl.174.software.zip)

1. Samuel Mensah, Kai Sun, Nikolaos Aletras. **An Empirical Study on Leveraging Position Embeddings for Target-oriented Opinion Words Extraction**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.722/) [[code]](https://github.com/samensah/Encoders_TOWE_EMNLP2021)

1. Amir Pouran Ben Veyseh, Nasim Nouri, Franck Dernoncourt, Dejing Dou, Thien Huu Nguyen. **Introducing Syntactic Structures into Target Opinion Word Extraction with Deep Learning**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.719.pdf)

1. Zhen Wu, Fei Zhao, Xin-Yu Dai, Shujian Huang, Jiajun Chen. **Latent Opinions Transfer Network for Target-Oriented Opinion Words Extraction**. AAAI 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6469/6325)

1. Zhifang Fan, Zhen Wu, Xin-Yu Dai, Shujian Huang, Jiajun Chen. **Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling**. NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1259.pdf)


### 1.7 Aspect-Opinion Pair Extraction

1. Shengqiong Wu, Hao Fei, Yafeng Ren, Donghong Ji, Jingye Li. **Learn from Syntax: Improving Pair-wise Aspect and Opinion Terms Extraction with Rich Syntactic Knowledge**. IJCAI 2021. [[paper]](https://www.ijcai.org/proceedings/2021/0545.pdf) [[code]](https://github.com/ChocoWu/Synfue-PAOTE)

1. Lei Gao, Yulong Wang, Tongcun Liu, Jingyu Wang, Lei Zhang, Jianxin Liao. **Question-Driven Span Labeling Model for Aspect–Opinion Pair Extraction**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17523)

1. Shaowei Chen, Jie Liu, Yu Wang, Wenzheng Zhang, Ziming Chi. **Synchronous Double-channel Recurrent Network for Aspect-Opinion Pair Extraction**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.582.pdf) [[code]](https://github.com/NKU-IIPLab/SDRN)

1. He Zhao, Longtao Huang, Rong Zhang, Quan Lu, Hui Xue. **SpanMlt: A Span-based Multi-Task Learning Framework for Pair-wise Aspect and Opinion Terms Extraction**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.296.pdf)


### 1.8 Aspect-Sentiment Pair Extraction

1. Ehsan Hosseini-Asl, Wenhao Liu, Caiming Xiong. **A Generative Language Model for Few-shot Aspect-Based Sentiment Analysis**. NAACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-naacl.58.pdf) [[code]](https://github.com/salesforce/fewshot_absa)

1. Lei Shu, Jiahua Chen, Bing Liu, Hu Xu. **Zero-Shot Aspect-Based Sentiment Analysis**. ArXiv 2022. [[paper]](https://arxiv.org/abs/2202.01924)

1. Wenxuan Zhang, Yang Deng, Xin Li, Lidong Bing, Wai Lam. **Aspect-based Sentiment Analysis in Question Answering Forums**. EMNLP Findings 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.390/) [[code]](https://github.com/IsakZhang/ABSA-QA)

1. Yunlong Liang, Fandong Meng, Jinchao Zhang, Yufeng Chen, Jinan Xu, Jie Zhou. **An Iterative Multi-Knowledge Transfer Network for Aspect-Based Sentiment Analysis**. EMNLP Findings 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.152/) [[code]](https://github.com/XL2248/IMKTN)

1. Guoxin Yu, Jiwei Li, Ling Luo, Yuxian Meng, Xiang Ao, Qing He. **Self Question-answering: Aspect-based Sentiment Analysis by Role Flipped Machine Reading Comprehension**. EMNLP Findings 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.115/)

1. Zeyu Li, Wei Cheng, Reema Kshetramade, John Houser, Haifeng Chen, Wei Wang. **Recommend for a Reason: Unlocking the Power of Unsupervised Aspect-Sentiment Co-Extraction**. EMNLP Findings 2021. [[paper]](https://aclanthology.org/2021.findings-emnlp.66/) [[code]](https://github.com/zyli93/ASPE-APRE)

1. Wenxuan Zhang, Ruidan He, Haiyun Peng, Lidong Bing, Wai Lam. **Cross-lingual Aspect-based Sentiment Analysis with Aspect Term Code-Switching**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.727/) [[code]](https://github.com/IsakZhang/XABSA)

1. Matan Orbach, Orith Toledo-Ronen, Artem Spector, Ranit Aharonov, Yoav Katz, Noam Slonim. **YASO: A Targeted Sentiment Analysis Evaluation Dataset for Open-Domain Reviews**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.721/) [[code]](https://github.com/IBM/yaso-tsa)

1. Jeremy Barnes, Robin Kurtz, Stephan Oepen, Lilja Øvrelid, Erik Velldal. **Structured Sentiment Analysis as Dependency Graph Parsing**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.263.pdf)

1. Shinhyeok Oh, Dongyub Lee, Taesun Whang, IlNam Park, Seo Gaeun, EungGyun Kim, Harksoo Kim. **Deep Context- and Relation-Aware Learning for Aspect-based Sentiment Analysis**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-short.63.pdf)

1. Rui Mao, Xiao Li. **Bridging Towers of Multi-task Learning with a Gating Mechanism for Aspect-based Sentiment Analysis and Sequential Metaphor Identification**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17596)

1. Yan Zhou, Fuqing Zhu, Pu Song, Jizhong Han, Tao Guo, Songlin Hu. **An Adaptive Hybrid Framework for Cross-domain Aspect-based Sentiment Analysis**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17719/17526)

1. Huaishao Luo, Lei Ji, Tianrui Li, Daxin Jiang, Nan Duan. **GRACE: Gradient Harmonized and Cascaded Labeling for Aspect-based Sentiment Analysis**. EMNLP 2020 Findings. [[paper]](https://aclanthology.org/2020.findings-emnlp.6/) [[code]](https://github.com/ArrowLuo/GRACE)

1. Chenggong Gong, Jianfei Yu, Rui Xia. **Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.572.pdf)

1. Jiaxin Huang, Yu Meng, Fang Guo, Heng Ji, Jiawei Han. **Weakly-Supervised Aspect-Based Sentiment Analysis via Joint Aspect-Sentiment Topic Embedding**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.568.pdf)

1. Zhuang Chen, Tieyun Qian. **Relation-Aware Collaborative Learning for Unified Aspect-Based Sentiment Analysis**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.340.pdf)

1. Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li and Yiwei Lv. **Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification**. ACL 2019. [[paper]](https://www.aclweb.org/anthology/P19-1051.pdf)

1. Huaishao Luo, Tianrui Li, Bing Liu and Junbo Zhang. **DOER: Dual Cross-Shared RNN for Aspect Term-Polarity Co-Extraction**. ACL 2019. [[paper]](https://www.aclweb.org/anthology/P19-1056.pdf)

1. Ruidan He, Wee Sun Lee, Hwee Tou Ng, and Daniel Dahlmeier. **An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis**. ACL 2019. [[paper]](https://www.aclweb.org/anthology/P19-1048.pdf)

1. Zheng Li, Xin Li, Ying Wei, Lidong Bing, Yu Zhang and Qiang Yang. **Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning**. EMNLP 2019. [[paper]](https://www.aclweb.org/anthology/D19-1466.pdf)

1. Xin Li, Lidong Bing, Piji Li, Wai Lam. **A Unified Model for Opinion Target Extraction and Target Sentiment Prediction**. AAAI 2019. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/download/4643/4521)

1. Feixiang Wang, Man Lan, Wenting Wang. **Towards a One-stop Solution to Both Aspect Extraction and Sentiment Analysis Tasks with Neural Multi-task Learning**. IJCNN 2018. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8489042)

1. Hao Li,Wei Lu. **Learning Latent Sentiment Scopes for Entity-Level Sentiment Analysis**. AAAI 2017. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14931/14137)

1. Meishan Zhang, Yue Zhang, Duy-Tin Vo. **Neural Networks for Open Domain Targeted Sentiment**. EMNLP 2015. [[paper]](https://aclanthology.org/D15-1073.pdf)

1. Margaret Mitchell, Jacqui Aguilar, Theresa Wilson, Benjamin Van Durme. **Open Domain Targeted Sentiment**. EMNLP 2013. [[paper]](https://aclanthology.org/D13-1171.pdf)

### 1.9 Category-Oriented Sentiment Classification


1. Jian Liu, Zhiyang Teng, Leyang Cui, Hanmeng Liu, Yue Zhang. **Solving Aspect Category Sentiment Analysis as a Text Generation Task**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.361/) [[code]](https://github.com/lgw863/ACSA-generation)

1. Bin Liang, Hang Su, Rongdi Yin, Lin Gui, Min Yang, Qin Zhao, Xiaoqi Yu, Ruifeng Xu. **Beta Distribution Guided Aspect-aware Graph for Aspect Category Sentiment Analysis with Affective Knowledge**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.19/) [[code]](https://github.com/BinLiang-NLP/AAGCN-ACSA)

1. Chi Sun, Luyao Huang, Xipeng Qiu. **Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence**. NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1035/) [[code]](https://github.com/HSLCY/ABSA-BERT-pair)

1. Bowen Xing, Lejian Liao, Dandan Song, Jingang Wang, Fuzhen Zhang, Zhongyuan Wang and Heyan Huang. **Earlier Attention? Aspect-Aware LSTM for Aspect-Based Sentiment Analysis**. IJCAI 2019. [[paper]](https://www.ijcai.org/proceedings/2019/0738.pdf)

1. Yukun Ma, Haiyun Peng, Erik Cambria. **Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM**. EMNLP 2018. [[paper]](https://ww.sentic.net/sentic-lstm.pdf)

1. Marzieh Saeidi, Guillaume Bouchard, Maria Liakata, Sebastian Riedel. **SentiHood: Targeted Aspect Based Sentiment Analysis Dataset for Urban Neighbourhoods**. COLING 2016. [[paper]](https://aclanthology.org/C16-1146.pdf)

1. Sebastian Ruder, Parsa Ghaffari, John G. Breslin. **A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis**. EMNLP 2016. [[paper]](https://aclanthology.org/D16-1103.pdf)

1. Caroline Brun, Diana Nicoleta Popa, Claude Roux. **XRCE: Hybrid Classification for Aspect-based Sentiment Analysis**. SemEval 2014. [[paper]](https://aclanthology.org/S14-2149.pdf)

### 1.10 Category-Sentiment Hierarchical Classification

1. Jiahao Bu, Lei Ren, Shuang Zheng, Yang Yang, Jingang Wang, Fuzheng Zhang, Wei Wu. **ASAP: A Chinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction**. NAACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.naacl-main.167.pdf)

1. Hongjie Cai, Yaofeng Tu, Xiangsheng Zhou, Jianfei Yu, Rui Xia. **Aspect-Category based Sentiment Analysis with Hierarchical Graph Convolutional Network**. COLING 2020. [[paper]](https://www.aclweb.org/anthology/2020.coling-main.72.pdf) [[code]](https://github.com/NUSTM/ACSA-HGCN)

1. Zehui Dai, Cheng Peng, Huajie Chen, Yadong Ding. **A Multi-Task Incremental Learning Framework with Category Name Embedding for Aspect-Category Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.565.pdf)

1. Yuncong Li, Cunxiang Yin, Sheng-hua Zhong, Xu Pan. **Multi-Instance Multi-Label Learning Networks for Aspect-Category Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.287.pdf)

1. Yuncong Li, Zhe Yang, Cunxiang Yin, Xu Pan, Lunan Cui, Qiang Huang, Ting Wei. **A Joint Model for Aspect-Category Sentiment Analysis with Shared Sentiment Prediction Layer**. CCL 2020. [[paper]](https://aclanthology.org/2020.ccl-1.103.pdf)

1. Martin Schmitt, Simon Steinheber, Konrad Schreiber, Benjamin Roth. **Joint Aspect and Polarity Classification for Aspect-based Sentiment Analysis with End-to-End Neural Networks**. EMNLP 2018. [[paper]](https://aclanthology.org/D18-1139.pdf)


### 1.11 Aspect-Category-Sentiment Triple Extraction

1. Hai Wan, Yufei Yang, Jianfeng Du, Yanan Liu, Kunxun Qi, Jeff Z. Pan. **Target-Aspect-Sentiment Joint Detection for Aspect-Based Sentiment Analysis**. AAAI 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6447/6303) [[code]](https://github.com/sysulic/TAS-BERT)

### 1.12 Aspect-Opinion-Sentiment Triple Extraction

1. Yue Mao, Yi Shen, Jingchao Yang, Xiaoying Zhu, Longjun Cai. **Seq2Path: Generating Sentiment Tuples as Paths of a Tree**. ACL Findings 2022. [[paper]](https://aclanthology.org/2022.findings-acl.174.pdf) [[code]](https://aclanthology.org/attachments/2022.findings-acl.174.software.zip)

1. Shu Liu, Kaiwen Li, Zuhe Li. **A Robustly Optimized BMRC for Aspect Sentiment Triplet Extraction**. NAACL 2022. [[paper]](https://aclanthology.org/2022.naacl-main.20.pdf)[[code]](https://github.com/itkaven/robmrc)

1. Hao Chen, Zepeng Zhai, Fangxiang Feng, Ruifan Li, Xiaojie Wang. **Enhanced Multi-Channel Graph Convolutional Network for Aspect Sentiment Triplet Extraction**. ACL 2022. [[paper]](https://aclanthology.org/2022.acl-long.212.pdf)[[code]](https://github.com/ccchenhao997/emcgcn-aste)

1. Hao Fei, Fei Li, Chenliang Li, Shengqiong Wu, Jingye Li, Donghong Ji. **Inheriting the Wisdom of Predecessors: A Multiplex Cascade Framework for Unified Aspect-based Sentiment Analysis**. IJCAI 2022 [[paper]](https://www.ijcai.org/proceedings/2022/0572.pdf)

1. Rajdeep Mukherjee, Tapas Nayak, Yash Butala, Sourangshu Bhattacharya, Pawan Goyal. **PASTE: A Tagging-Free Decoding Framework Using Pointer Networks for Aspect Sentiment Triplet Extraction**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.731/) [[code]](https://github.com/rajdeep345/PASTE/)

1. Hongjiang Jing, Zuchao Li, Hai Zhao, Shu Jiang. **Seeking Common but Distinguishing Difference, A Joint Aspect-based Sentiment Analysis Model**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.318/) 

1. Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing, Wai Lam. **Towards Generative Aspect-Based Sentiment Analysis**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-short.64/) [[code]](https://github.com/IsakZhang/Generative-ABSA)

1. Lu Xu, Yew Ken Chia, Lidong Bing. **Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.367/) [[code]](https://github.com/chiayewken/Span-ASTE)

1. Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu and Zheng Zhang. **A Unified Generative Framework for Aspect-Based Sentiment Analysis**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.188/) [[code]](https://github.com/yhcc/BARTABSA)

1. Shaowei Chen, Yu Wang, Jie Liu, Yuelin Wang. **Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17500/17307)

1. Yue Mao, Yi Shen, Chao Yu, Longjun Cai. **A Joint Training Dual-MRC Framework for Aspect Based Sentiment Analysis**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17597/17404)

1. Lu Xu, Hao Li, Wei Lu, Lidong Bing. **Position-Aware Tagging for Aspect Sentiment Triplet Extraction**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.183.pdf)

1. Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, Rui Xia. **Grid Tagging Scheme for End-to-End Fine-grained Opinion Extraction**. EMNLP 2020, Findings. [[paper]](https://www.aclweb.org/anthology/2020.findings-emnlp.234.pdf) [[code]](https://github.com/NJUNLP/GTS)

1. Chen Zhang, Qiuchi Li, Dawei Song, Benyou Wang. **A Multi-task Learning Framework for Opinion Triplet Extraction**. EMNLP 2020, Findings. [[paper]](https://aclanthology.org/2020.findings-emnlp.72) [[code]](https://github.com/GeneZC/OTE-MTL)

1. Haiyun Peng, Lu Xu, Lidong Bing, Wei Lu, Fei Huang. **Knowing What, How and Why: A Near Complete Solution for Aspect-based Sentiment Analysis**. AAAI 2020. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/download/6383/6239) [[data]](https://github.com/xuuuluuu/SemEval-Triplet-data)

1. Minqing Hu, Bing Liu. **Mining and Summarizing Customer Reviews**. KDD 2004. [[paper]](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9730&rep=rep1&type=pdf)

### 1.13 Aspect-Category-Opinion-Sentiment Quadruple Extraction

1. Yue Mao, Yi Shen, Jingchao Yang, Xiaoying Zhu, Longjun Cai. **Seq2Path: Generating Sentiment Tuples as Paths of a Tree**. ACL  Findings 2022. [[paper]](https://aclanthology.org/2022.findings-acl.174.pdf)[[code]](https://aclanthology.org/attachments/2022.findings-acl.174.software.zip)

1. Xiaoyi Bao, Wang Zhongqing, Xiaotong Jiang, Rong Xiao, Shoushan Li. **Aspect-based Sentiment Analysis with Opinion Tree Generation**. IJCAI 2022. [[paper]](https://www.ijcai.org/proceedings/2022/0561.pdf)

1. Wenxuan Zhang, Yang Deng, Xin Li, Yifei Yuan, Lidong Bing, Wai Lam. **Aspect Sentiment Quad Prediction as Paraphrase Generation**. EMNLP 2021. [[paper]](https://aclanthology.org/2021.emnlp-main.726/) [[code]](https://github.com/IsakZhang/ABSA-QUAD)

1. Hongjie Cai, Rui Xia, Jianfei Yu. **Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.29.pdf) [[code & data]](https://github.com/NUSTM/ACOS) ![](https://img.shields.io/badge/-The%20first%20work%20introducing%20the%20ABSA%20Quadruple%20Extraction%20task-red)
 
 ## 2. Cross-Domain ABSA
 
 ### 2.1 Cross-Domain Aspect Extraction
1. Lekhtman, Entony and Ziser, Yftah and Reichart, Roi. **DILBERT: Customized Pre-Training for Domain Adaptation with Category Shift, with an Application to Aspect Extraction**. EMNLP 2021.[[paper]](https://aclanthology.org/2021.emnlp-main.20.pdf)
1. Zhuang Chen and Tieyun Qian. **Bridge-Based Active Domain Adaptation for Aspect Term Extraction**. ACL 2021. [[paper]](https://aclanthology.org/2021.acl-long.27.pdf)
1. Tao Liang, Wenya Wang and Fengmao Lv. **Weakly Supervised Domain Adaptation for Aspect Extraction via Multi-level Interaction Transfer**. IEEE TNNLS 2021. [[paper]](https://arxiv.org/abs/2006.09235)
1. Ying Ding, Jianfei Yu, and Jing Jiang. **Recurrent Neural Networks with Auxiliary Labels for Cross Domain Opinion Target Extraction**. AAAI 2017. [[paper]](https://www.semanticscholar.org/paper/Recurrent-Neural-Networks-with-Auxiliary-Labels-for-Ding-Yu/d08341562091ac6777f613a68a0d59eb600b5c57)


### 2.2 Cross-Domain Aspect-Opinion Co-Extraction
1. Oren Pereg, Daniel Korat, and Moshe Wasserblat. **Syntactically Aware Cross-Domain Aspect and Opinion Terms Extraction**. COLING 2020. [[paper]](https://aclanthology.org/2020.coling-main.158.pdf)
1. Wenya Wang and Sinno Jialin Pan. **Syntactically-Meaningful and Transferable Recursive Neural Networks for Aspect and Opinion Extraction**. Computational Linguistics (CL) 2019. [[paper]](https://aclanthology.org/J19-4004.pdf)
1. Wenya Wang and Sinno Jialin Pan. **Transferable Interactive Memory Network for Domain Adaptation in Fine-grained Opinion Extraction**. AAAI 2019. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4703/4581)
1. Wenya Wang and Sinno Jialin Pan.  **Recursive Neural Structural Correspondence Network for Cross-Domain Aspect and Opinion Co-Extraction**. ACL 2018. [[paper]](https://www.aclweb.org/anthology/P18-1202.pdf)



### 2.3 Cross-Domain Aspect-Oriented Sentiment Classification
1. Kai Zhang, Qi Liu, Hao Qian, Biao Xiang, Qing Cui, Jun Zhou, and Enhong Chen. **EATN: An Efficient Adaptive Transfer Network for Aspect-level Sentiment Analysis**. TKDE 2021. [[paper]](https://ieeexplore.ieee.org/abstract/document/9415156)
1. Mengting Hu, Yike Wu, Shiwan Zhao, Honglei Guo, Renhong Cheng, and Zhong Su. **Domain-Invariant Feature Distillation for Cross-Domain Sentiment Classification**. EMNLP-IJCNLP 2019. [[paper]](https://aclanthology.org/D19-1558.pdf)


### 2.4 Cross-Domain Aspect-Sentiment Pair Extraction
1. Jianfei Yu, Chenggong Gong, and Rui Xia. **Cross-Domain Review Generation for Aspect-Based Sentiment Analysis**. ACL 2021，Findings. [[paper]](https://aclanthology.org/2021.findings-acl.421.pdf) [[code]](https://github.com/NUSTM/CDRG)
1. Yan Zhou, Fuqing Zhu, Pu Song, Jizhong Han, Tao Guo, and Songlin Hu. **An Adaptive Hybrid Framework for Cross-domain Aspect-based Sentiment Analysis**. AAAI 2021. [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17719)
1. Chenggong Gong, Jianfei Yu, and Rui Xia. **Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis**. EMNLP 2020. [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.572.pdf) [[code]](https://github.com/NUSTM/BERT-UDA)
1. Zheng Li, Xin Li, Ying Wei, Lidong Bing, Yu Zhang and Qiang Yang. **Transferable End-to-End Aspect-based Sentiment Analysis with Selective Adversarial Learning**. EMNLP 2019. [[paper]](https://arxiv.org/abs/1910.14192) 
 
 ## 3. Multi-Modal ABSA
 
### 3.1 Multi-Modal Aspect Extraction (& Multi-Modal Named Entity Recognition)
1. Xinyu Wang, Jiong Cai, Yong Jiang, Pengjun Xie, Kewei Tu and Wei Lu. **Named Entity and Relation Extraction with Multi-Modal Retrieval**. EMNLP 2022 Findings
1. Baohang Zhou, Ying Zhang, Kehui Song, Wenya Guo, Guoqing Zhao, hongbin wang and Xiaojie Yuan. **A Span-based Multimodal Variational Autoencoder for Semi-supervised Multimodal Named Entity Recognition**. EMNLP 2022 [[code]](https://github.com/ZovanZhou/SMVAE)
1. Gang Zhao, Guanting Dong, Yidong Shi, Haolong Yan, Weiran Xu and Si Li. **Entity-level Interaction via Heterogeneous Graph for Multimodal Named Entity Recognition**. EMNLP 2022 Findings 
2. Jie Wang, Yan Yang, Keyu Liu, Zhiping Zhu and Xiaorong Liu. **M3S: Scene graph driven Multi-granularity Multi-task learning for Multi-modal NER**. TASLP 2022 [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9944151)
3. Xuwu Wang; Jiabo Ye; Zhixu Li; Junfeng Tian; Yong Jiang; Ming Yan; Ji Zhang; Yanghua Xiao. **CAT-MNER: Multimodal Named Entity Recognition with Knowledge-Refined Cross-Modal Attention**. ICME 2022 [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9859972)
4. Junyu Lu, Dixiang Zhang, Jiaxing Zhang, Pingjian Zhang. **Flat Multi-modal Interaction Transformer for Named Entity Recognition**. COLING 2022 [[paper]](https://aclanthology.org/2022.coling-1.179.pdf)
5. Bo Xu, Shizhou Huang, Ming Du, Hongya Wang, Hui Song. **Different Data, Different Modalities! Reinforced Data Splitting for Effective Multimodal Information Extraction from Social Media Posts**. COLING 2022 [[paper]](https://aclanthology.org/2022.coling-1.160.pdf)
6.  Meihuizi Jia, Xin Shen, Lei Shen, Jinhui Pang, Lejian Liao, Yang Song, Meng Chen, Xiaodong He. **Query Prior Matters: A MRC Framework for Multimodal Named Entity Recognition**. ACM MM 2022 [[paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548427)
7.  Fei Zhao, Chunhui Li, Zhen Wu, Shangyu Xing, Xinyu Dai. **Learning from Different text-image Pairs: A Relation-enhanced Graph Convolutional Network for Multimodal NER**. ACM MM 2022 [[paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548228)
1. Xiang Chen, Ningyu Zhang, Lei Li, Shumin Deng, Chuanqi Tan, Changliang Xu, Fei Huang, Luo Si, and Huajun Chen. **Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion**. SIGIR 2022. [[paper]](https://arxiv.org/pdf/2205.02357.pdf) [[code]](https://github.com/zjunlp/MKGformer)
2. Xinyu Wang, Min Gui, Yong Jiang, Zixia Jia, Nguyen Bach, Tao Wang, Zhongqiang Huang, Fei Huang, and Kewei Tu. **ITA: Image-Text Alignments for Multi-Modal Named Entity Recognition**. NAACL 2022. [[paper]](https://arxiv.org/pdf/2112.06482.pdf) [[code]](https://github.com/Alibaba-NLP/KB-NER/tree/main/ITA)
3. Xiang Chen, Ningyu Zhang, Lei Li, Shumin Deng, Chuanqi Tan, Changliang Xu, Fei Huang, Luo Si, and Huajun Chen. **Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction**. NAACL 2022 Findings. [[paper]](https://arxiv.org/pdf/2205.03521.pdf) [[code]](https://github.com/zjunlp/HVPNeT)
4. Bo Xu, Shizhou Huang, Chaofeng Sha, and Hongya Wang. **MAF: A General Matching and Alignment Framework for Multimodal Named Entity Recognition**. WSDM 2022. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3488560.3498475?casa_token=mgnoLxqNY6sAAAAA:V6QKO_hH7_RkHeeDThokhq6vgRFdRqdepH9ZTPt5Ft1T9Qmj-KK4HPkoWI0TDn1I4nf-K15EaScVOg)
5. Dong Zhang, Suzhong Wei, Shoushan Li, Hanqian Wu, Qiaoming Zhu, and Guodong Zhou. **Multi-modal Graph Fusion for Named Entity Recognition with Targeted Visual Guidance**. AAAI 2021. [[paper]](https://www.aaai.org/AAAI21Papers/AAAI-2753.ZhangD.pdf) [[code]](https://github.com/TransformersWsz/UMGF)
6. Lin Sun, Jiquan Wang, Kai Zhang, Yindu Su and Fangsheng Weng. **RpBERT: A Text-image Relation Propagation-based BERT Model for Multimodal NER**. AAAI 2021. [[paper]](https://www.aaai.org/AAAI21Papers/AAAI-761.SunL.pdf) [[code]](https://github.com/Multimodal-NER/RpBERT)
7. Dawei Chen, Zhixu Li, Binbin Gu, and Zhigang Chen. **Multimodal Named Entity Recognition with Image Attributes and Image Knowledge**. DASFAA 2021. [[paper]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-73197-7_12.pdf)
8. Shuguang Chen, Gustavo Aguilar, Leonardo Neves and Thamar Solorio. **Can images help recognize entities? A study of the role of images for Multimodal NER**. WNUT 2021. [[paper]](https://aclanthology.org/2021.wnut-1.11.pdf)
9. Luping Liu, Meiling Wang, Mozhi Zhang, Linbo Qing and Xiaohai He. **UAMNer: uncertainty-aware multimodal named entity recognition in social media posts**. Applied Intelligence 2021. [[paper]](https://link.springer.com/content/pdf/10.1007/s10489-021-02546-5.pdf)
10. Zhiwei Wu, Changmeng Zheng, Yi Cai, Junying Chen, Ho-fung Leung and Qing Li. **Multimodal Representation with Embedded Visual Guiding Objects for Named Entity Recognition in Social Media Posts**. ACM MM 2020. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3394171.3413650)
11. Changmeng Zheng, Zhiwei Wu, Tao Wang, Yi Cai and Qing Li. **Object-Aware Multimodal Named Entity Recognition in Social Media Posts With Adversarial Learning**. IEEE Transactions on Multimedia 2020. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9154571)
12. Jianfei Yu, Jing Jiang, Li Yang, and Rui Xia. **Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer**. ACL 2020. [[paper]](https://aclanthology.org/2020.acl-main.306.pdf) [[code]](https://github.com/jefferyYu/UMT)
13. Hanqian Wu, Siliang Cheng, Jingjing Wang, Shoushan Li, and Lian Chi. **Multimodal Aspect Extraction with Region-Aware Alignment Network**. NLPCC 2020. [[paper]](https://www.springerprofessional.de/en/multimodal-aspect-extraction-with-region-aware-alignment-network/18449698)
14. Di Lu, Leonardo Neves, Vitor Carvalho, Ning Zhang and Heng Ji. **Visual Attention Model for Name Tagging in Multimodal Social Media**. ACL 2018. [[paper]](https://aclanthology.org/P18-1185.pdf)
15. Qi Zhang, Jinlan Fu, Xiaoyu Liu and Xuanjing Huang. **Adaptive Co-Attention Network for Named Entity Recognition in Tweets**. AAAI 2018. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16432/16127)
16. Seungwhan Moon, Leonardo Neves and Vitor Carvalho. **Multimodal Named Entity Recognition for Short Social Media Posts**. NAACL 2018. [[paper]](https://aclanthology.org/N18-1078.pdf)


### 3.2 Multi-Modal Category-Oriented Sentiment Classification
1. Jianfei Yu, Kai Chen, and Rui Xia. **Hierarchical Interactive Multimodal Transformer for Aspect-Based Multimodal Sentiment Analysis**. IEEE Transactions on Affective Computing 2022. [[paper]](https://ieeexplore.ieee.org/abstract/document/9765342)
2. Jie Zhou, Jiabao Zhao, Jimmy Xiangji Huang, Qinmin Vivian Hu, and Liang He. **MASAD: A Large-Scale Dataset for Multimodal Aspect-Based Sentiment Analysis**. Neurocomputing 2021. [[paper]](https://www.sciencedirect.com/science/article/pii/S0925231221007931)
3. Nan Xu, Wenji Mao, and Guandan Chen. **Multi-interactive Memory Network for Aspect Based Multimodal Sentiment Analysis**. AAAI 2019. [[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/3807) [[code]](https://github.com/xunan0812/MIMN)



### 3.3 Multi-Modal Aspect-Oriented Sentiment Classification
1. Hao Yang, Yanyan Zhao and Bing Qin. **Face-Sensitive Image-to-Emotional-Text Cross-modal Translation for Multimodal Aspect-based Sentiment Analysis**. EMNLP 2022 [[code]](https://github.com/yhit98/FITE)
1. Fei Zhao, Zhen Wu, Siyu Long, Xinyu Dai, Shujian Huang, Jiajun Chen. **Learning from Adjective-Noun Pairs: A Knowledge-enhanced Framework for Target-Oriented Multimodal Sentiment Classification**. COLING 2022 [[paper]](https://aclanthology.org/2022.coling-1.590.pdf)
1. Yufeng Huang, Zhuo Chen, Wen Zhang, Jiaoyan Chen, Jeff Z. Pan,Zhen Yao, Yujie Xie, Huajun Chen. **Aspect-based Sentiment Classification with Sequential Cross-modal Semantic Graph**. arxiv 2022 [[paper]](https://arxiv.org/pdf/2208.09417.pdf)
1. Yang Yu, Dong Zhang, Shoushan Li. **Unified Multi-modal Pre-training for Few-shot Sentiment Analysis with Prompt-based Learning**. ACM MM2022 [[paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548306)
3. Junjie Ye, Jie Zhou, Junfeng Tian, Rui Wang, Jingyi Zhou, Tao Gui, Qi Zhang,Xuanjing Huang. **Sentiment-aware multimodal pre-training for multimodal sentiment
analysis**. KBS 2022 [[paper]](https://www.sciencedirect.com/science/article/pii/S0950705122011145)
1. Zhen Li, Bing Xu, Conghui Zhu, and Tiejun Zhao. **CLMLF: A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection**. NAACL 2022 Findings. [[paper]](https://aclanthology.org/2022.findings-naacl.175.pdf)
2. Jianfei Yu, Jieming Wang, Rui Xia and Junjie Li. **Targeted Multimodal Sentiment Classification based on Coarse-to-Fine Grained Image-Target Matching**. IJCAI 2022. [[paper]](https://www.ijcai.org/proceedings/2022/0622.pdf)
3. Zaid Khan and Yun Fu. **Exploiting BERT For Multimodal Target Sentiment Classification Through Input Space Translation**. ACM MM 2021 [[paper]](https://arxiv.org/pdf/2108.01682.pdf) [[code]](https://github.com/codezakh/exploiting-BERT-thru-translation)
4. Zhe Zhang, Zhu Wang, Xiaona Li, Nannan Liu, Bin Guo, and Zhiwen Yu. **ModalNet: an aspect-level sentiment classification model by exploring multimodal data with fusion discriminant attentional network**. World Wide Web 2021. [[paper]](https://link.springer.com/article/10.1007/s11280-021-00955-7) 
5. Jianfei Yu, Jing Jiang and Rui Xia. **Entity-Sensitive Attention and Fusion Network for Entity-Level Multimodal Sentiment Classification**. IEEE/ACM TASLP 2020. [[paper]](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6507&context=sis_research)
6. Jianfei Yu and Jing Jiang. **Adapting BERT for Target-Oriented Multimodal Sentiment Classification**. IJCAI 2019. [[paper]](https://www.ijcai.org/Proceedings/2019/0751.pdf) [[code]](https://github.com/jefferyYu/TomBERT)

### 3.4 Multi-Modal Aspect-Sentiment Pair Extraction
1. Ru Zhou, Wenya Guo, Xumeng Liu, Shenglong Yu, Ying Zhang, and Xiaojie Yuan. AoM: Detecting Aspect-oriented Information for Multimodal Aspect-Based Sentiment Analysis. Findings of ACL 2023. [[paper]](https://arxiv.org/pdf/2306.01004.pdf) [[code]](https://github.com/SilyRab/AoM)
2. Xiaocui Yang, Shi Feng, Daling Wang, Sun Qi, Wenfang Wu, Yifei Zhang, Pengfei Hong, and Soujanya Poria.Few-shot Joint Multimodal Aspect-Sentiment Analysis Based on Generative Multimodal Prompt. Findings of ACL 2023. [[paper]](https://arxiv.org/pdf/2305.10169.pdf) [[code]](https://github.com/YangXiaocui1215/GMP)
3. Zhewen Yu, Jin Wang, Liang-Chih Yu, and Xuejie Zhang. Dual-Encoder Transformers with Cross-modal Alignment for Multimodal Aspect-based Sentiment Analysis. AACL-IJCNLP 2022. [[paper]](https://aclanthology.org/2022.aacl-main.32.pdf) [[code]](https://github.com/windforfurture/DTCA)
4. Li Yang, Jin-Cheon Na, and Jianfei Yu. Cross-Modal Multitask Transformer for End-to-End Multimodal Aspect-Based Sentiment Analysis. Information Processing and Management, 59(5), 103038, 2022. [[paper]](https://www.sciencedirect.com/science/article/pii/S0306457322001479?via%3Dihub) [[code]](https://github.com/yangli-hub/CMMT-Code)
5. Yan Ling, Jianfei Yu, and Rui Xia. **Vision-Language Pre-Training for Multimodal Aspect-Based Sentiment Analysis**. ACL 2022. [[paper]](https://aclanthology.org/2022.acl-long.152.pdf) [[code]](https://github.com/NUSTM/VLP-MABSA)
6. Xincheng Ju,  Dong Zhang,  Rong Xiao,  Junhui Li, Shoushan Li, Min Zhang, and Guodong Zhou. **Joint Multi-Modal Aspect-Sentiment Analysis with Auxiliary Cross-Modal Relation  Detection**. EMNLP 2021 [[paper]](https://aclanthology.org/2021.emnlp-main.360.pdf) [[code]](https://github.com/MANLP-suda/JML)
