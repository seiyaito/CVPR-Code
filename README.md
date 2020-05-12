# CVPR2020-Code

CVPR 2020の論文及びオープンソースプロジェクトのコレクションです.そして,もし問題があればissueに提出して,CVPR 2020のオープンソースプロジェクトを共有することも歓迎します

- [CNN](#CNN)
- [画像分類](#Image-Classification)
- [物体検出](#Object-Detection)
- [3D物体検出](#3D-Object-Detection)
- [動画物体検出](#Video-Object-Detection)
- [物体追跡](#Object-Tracking)
- [セマンティックセグメンテーション](#Semantic-Segmentation)
- [インスタンスセグメンテーション](#Instance-Segmentation)
- [全景分割](#Panoptic-Segmentation)
- [動画物体分割](#VOS)
- [スーパーピクセルセグメンテーション](#Superpixel)
- [NAS](#NAS)
- [GAN](#GAN)
- [Re-ID](#Re-ID)
- [3D点群（分類/分割/レジストレーション等）](#3D-PointCloud)
- [顔（識別/検出/重建等）](#Face)
- [人体姿勢推定(2D/3D)](#Human-Pose-Estimation)
- [人体解析](#Human-Parsing)
- [シーンテキスト検出](#Scene-Text-Detection)
- [シーンテキスト識別](#Scene-Text-Recognition)
- [超解像](#Super-Resolution)
- [モデル圧縮/枝刈り](#Model-Compression)
- [動画理解/行動識別](#Action-Recognition)
- [群衆カウント](#Crowd-Counting)
- [深度推定](#Depth-Estimation)
- [6D物体姿勢推定](#6DOF)
- [手姿勢推定](#Hand-Pose)
- [显著性検出](#Saliency)
- [画像修復](#Denoising)
- [デブラー](#Deblurring)
- [去雾](#Dehazing)
- [特徴点検出・記述](#Feature)
- [VQA(VQA)](#VQA)
- [動画问答(VideoQA)](#VideoQA)
- [視覚言語ナビゲーション](#VLN)
- [動画圧縮](#Video-Compression)
- [動画補間](#Video-Frame-Interpolation)
- [スタイル変換](#Style-Transfer)
- [车道线検出](#Lane-Detection)
- [Human-Object Interaction (HOI)検出](#HOI)
- [行動軌跡予測](#HTP)
- [モーション予測](#Motion-Predication)
- [虚拟试衣](#Virtual-Try-On)
- [HDR](#HDR)
- [对抗样本](#AE)
- [データセット](#Datasets)
- [その他](#Others)
- [採択されたか不明](#Not-Sure)

<a name="CNN"></a>

# CNN

**Exploring Self-attention for Image Recognition**

- 論文：https://hszhao.github.io/papers/cvpr20_san.pdf

- コード：https://github.com/hszhao/SAN

**Improving Convolutional Networks with Self-Calibrated Convolutions**

- ホームページ：https://mmcheng.net/scconv/

- 論文：http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
- コード：https://github.com/backseason/SCNet

**Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets**

- 論文：https://arxiv.org/abs/2003.13549
- コード：https://github.com/zeiss-microscopy/BSConv

<a name="Image-Classification"></a>

# 画像分類

**Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion**

- 論文：https://arxiv.org/abs/2003.04490

- コード：https://github.com/AdamKortylewski/CompositionalNets

**Spatially Attentive Output Layer for Image Classification**

- 論文：https://arxiv.org/abs/2004.07570 

- コード（好像被原作者删除了）：https://github.com/ildoonet/spatially-attentive-output-layer 

<a name="Object-Detection"></a>

# 物体検出

**Scale-Equalizing Pyramid Convolution for Object Detection**

論文：https://arxiv.org/abs/2005.03101

コード：https://github.com/jshilong/SEPC

**Revisiting the Sibling Head in Object Detector**

- 論文：https://arxiv.org/abs/2003.07540

- コード：https://github.com/Sense-X/TSD 

**Scale-equalizing Pyramid Convolution for Object Detection**

- 論文：なし
- コード：https://github.com/jshilong/SEPC 

**Detection in Crowded Scenes: One Proposal, Multiple Predictions**

- 論文：https://arxiv.org/abs/2003.09163
- コード：https://github.com/megvii-model/CrowdDetection

**Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection**

- 論文：https://arxiv.org/abs/2004.04725
- コード：https://github.com/NVlabs/wetectron

**Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection**

- 論文：https://arxiv.org/abs/1912.02424 
- コード：https://github.com/sfzhang15/ATSS

**BiDet: An Efficient Binarized Object Detector**

- 論文：https://arxiv.org/abs/2003.03961 
- コード：https://github.com/ZiweiWangTHU/BiDet

**Harmonizing Transferability and Discriminability for Adapting Object Detectors**

- 論文：https://arxiv.org/abs/2003.06297
- コード：https://github.com/chaoqichen/HTCN

**CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection**

- 論文：https://arxiv.org/abs/2003.09119
- コード：https://github.com/KiveeDong/CentripetalNet

**Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection**

- 論文：https://arxiv.org/abs/2003.11818
- コード：https://github.com/ggjy/HitDet.pytorch

**EfficientDet: Scalable and Efficient Object Detection**

- 論文：https://arxiv.org/abs/1911.09070
- コード：https://github.com/google/automl/tree/master/efficientdet 

<a name="3D-Object-Detection"></a>

# 3D物体検出

**MLCVNet: Multi-Level Context VoteNet for 3D Object Detection**

- 論文：https://arxiv.org/abs/2004.05679
- コード：https://github.com/NUAAXQ/MLCVNet

**3DSSD: Point-based 3D Single Stage Object Detector**

- CVPR 2020 Oral

- 論文：https://arxiv.org/abs/2002.10187

- コード：https://github.com/tomztyang/3DSSD

**Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation**

- 論文：https://arxiv.org/abs/2004.03572

- コード：https://github.com/zju3dv/disprcn

**End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection**

- 論文：https://arxiv.org/abs/2004.03080

- コード：https://github.com/mileyan/pseudo-LiDAR_e2e

**DSGN: Deep Stereo Geometry Network for 3D Object Detection**

- 論文：https://arxiv.org/abs/2001.03398
- コード：https://github.com/chenyilun95/DSGN

**LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention**

- 論文：https://arxiv.org/abs/2004.01389
- コード：https://github.com/yinjunbo/3DVID

**PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection**

- 論文：https://arxiv.org/abs/1912.13192

- コード：https://github.com/sshaoshuai/PV-RCNN

**Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud**

- 論文：https://arxiv.org/abs/2003.01251 
- コード：https://github.com/WeijingShi/Point-GNN 

<a name="Video-Object-Detection"></a>

# 動画物体検出

**Memory Enhanced Global-Local Aggregation for Video Object Detection**

論文：https://arxiv.org/abs/2003.12063

コード：https://github.com/Scalsol/mega.pytorch

<a name="Object-Tracking"></a>

# 物体追跡

**D3S -- A Discriminative Single Shot Segmentation Tracker**

- 論文：https://arxiv.org/abs/1911.08862
- コード：https://github.com/alanlukezic/d3s

**ROAM: Recurrently Optimizing Tracking Model**

- 論文：https://arxiv.org/abs/1907.12006

- コード：https://github.com/skyoung/ROAM

**Siam R-CNN: Visual Tracking by Re-Detection**

- ホームページ：https://www.vision.rwth-aachen.de/page/siamrcnn
- 論文：https://arxiv.org/abs/1911.12836
- 論文2：https://www.vision.rwth-aachen.de/media/papers/192/siamrcnn.pdf
- コード：https://github.com/VisualComputingInstitute/SiamR-CNN

**Cooling-Shrinking Attack: Blinding the Tracker with Imperceptible Noises**

- 論文：https://arxiv.org/abs/2003.09595 
- コード：https://github.com/MasterBin-IIAU/CSA 

**High-Performance Long-Term Tracking with Meta-Updater**

- 論文：https://arxiv.org/abs/2004.00305

- コード：https://github.com/Daikenan/LTMU

**AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization**

- 論文：https://arxiv.org/abs/2003.12949

- コード：https://github.com/vision4robotics/AutoTrack

**Probabilistic Regression for Visual Tracking**

- 論文：https://arxiv.org/abs/2003.12565
- コード：https://github.com/visionml/pytracking

**MAST: A Memory-Augmented Self-supervised Tracker**

- 論文：https://arxiv.org/abs/2002.07793
- コード：https://github.com/zlai0/MAST

**Siamese Box Adaptive Network for Visual Tracking**

- 論文：https://arxiv.org/abs/2003.06761

- コード：https://github.com/hqucv/siamban

<a name="Semantic-Segmentation"></a>

# セマンティックセグメンテーション

**Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation**

- 論文：https://arxiv.org/abs/2003.00867
- コード：https://github.com/MyeongJin-Kim/Learning-Texture-Invariant-Representation

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**

- 論文：http://vladlen.info/papers/MSeg.pdf
- コード：https://github.com/mseg-dataset/mseg-api

**CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement**

- 論文：https://arxiv.org/abs/2005.02551
- コード：https://github.com/hkchengrex/CascadePSP

**Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision**

- Oral
- 論文：https://arxiv.org/abs/2004.07703
- コード：https://github.com/feipan664/IntraDA

**Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**

- 論文：https://arxiv.org/abs/2004.04581
- コード：https://github.com/YudeWang/SEAM

**Temporally Distributed Networks for Fast Video Segmentation**

- 論文：https://arxiv.org/abs/2004.01800

- コード：https://github.com/feinanshan/TDNet

**Context Prior for Scene Segmentation**

- 論文：https://arxiv.org/abs/2004.01547

- コード：https://git.io/ContextPrior

**Strip Pooling: Rethinking Spatial Pooling for Scene Parsing**

- 論文：https://arxiv.org/abs/2003.13328

- コード：https://github.com/Andrew-Qibin/SPNet

**Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**

- 論文：https://arxiv.org/abs/2003.05128
- コード：https://github.com/shachoi/HANet

**Learning Dynamic Routing for Semantic Segmentation**

- 論文：https://arxiv.org/abs/2003.10401

- コード：https://github.com/yanwei-li/DynamicRouting

<a name="Instance-Segmentation"></a>

# インスタンスセグメンテーション

**PolarMask: Single Shot Instance Segmentation with Polar Representation**

- 論文：https://arxiv.org/abs/1909.13226 
- コード：https://github.com/xieenze/PolarMask 
- 解説：https://zhuanlan.zhihu.com/p/84890413 

**CenterMask : Real-Time Anchor-Free Instance Segmentation**

- 論文：https://arxiv.org/abs/1911.06667 
- コード：https://github.com/youngwanLEE/CenterMask 

**BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation**

- 論文：https://arxiv.org/abs/2001.00309
- コード：https://github.com/aim-uofa/AdelaiDet

**Deep Snake for Real-Time Instance Segmentation**

- 論文：https://arxiv.org/abs/2001.01629
- コード：https://github.com/zju3dv/snake

**Mask Encoding for Single Shot Instance Segmentation**

- 論文：https://arxiv.org/abs/2003.11712

- コード：https://github.com/aim-uofa/AdelaiDet

<a name="Panoptic-Segmentation"></a>

# 全景分割

**Pixel Consensus Voting for Panoptic Segmentation**

- 論文：https://arxiv.org/abs/2004.01849
- コード：还未公布

**BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation**

論文：https://arxiv.org/abs/2003.14031

コード：https://github.com/Mooonside/BANet

<a name="VOS"></a>

# 動画物体分割

**A Transductive Approach for Video Object Segmentation**

- 論文：https://arxiv.org/abs/2004.07193

- コード：https://github.com/microsoft/transductive-vos.pytorch

**State-Aware Tracker for Real-Time Video Object Segmentation**

- 論文：https://arxiv.org/abs/2003.00482

- コード：https://github.com/MegviiDetection/video_analyst

**Learning Fast and Robust Target Models for Video Object Segmentation**

- 論文：https://arxiv.org/abs/2003.00908 
- コード：https://github.com/andr345/frtm-vos

**Learning Video Object Segmentation from Unlabeled Videos**

- 論文：https://arxiv.org/abs/2003.05020
- コード：https://github.com/carrierlxk/MuG

<a name="Superpixel"></a>

# スーパーピクセルセグメンテーション

**Superpixel Segmentation with Fully Convolutional Networks**

- 論文：https://arxiv.org/abs/2003.12929
- コード：https://github.com/fuy34/superpixel_fcn

<a name="NAS"></a>

# NAS

**Densely Connected Search Space for More Flexible Neural Architecture Search**

- 論文：https://arxiv.org/abs/1906.09607

- コード：https://github.com/JaminFong/DenseNAS

**MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning**

- 論文：https://arxiv.org/abs/2003.14058

- コード：https://github.com/bhpfelix/MTLNAS

**FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions**

- 論文下载链接：https://arxiv.org/abs/2004.05565

- コード：https://github.com/facebookresearch/mobile-vision

**Neural Architecture Search for Lightweight Non-Local Networks**

- 論文：https://arxiv.org/abs/2004.01961
- コード：https://github.com/LiYingwei/AutoNL

**Rethinking Performance Estimation in Neural Architecture Search**

- 論文：準備中
- コード：https://github.com/zhengxiawu/rethinking_performance_estimation_in_NAS
- 解説：https://www.zhihu.com/question/372070853/answer/1035234510

**CARS: Continuous Evolution for Efficient Neural Architecture Search**

- 論文：https://arxiv.org/abs/1909.04977 
- コード（公開前）：https://github.com/huawei-noah/CARS 

<a name="GAN"></a>

# GAN

**Semantically Mutil-modal Image Synthesis**

- ホームページ：http://seanseattle.github.io/SMIS
- 論文：https://arxiv.org/abs/2003.12697
- コード：https://github.com/Seanseattle/SMIS

**Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping**

- 論文：https://yiranran.github.io/files/CVPR2020_Unpaired%20Portrait%20Drawing%20Generation%20via%20Asymmetric%20Cycle%20Mapping.pdf
- コード：https://github.com/yiranran/Unpaired-Portrait-Drawing

**Learning to Cartoonize Using White-box Cartoon Representations**

- 論文：https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf

- ホームページ：https://systemerrorwang.github.io/White-box-Cartoonization/
- コード：https://github.com/SystemErrorWang/White-box-Cartoonization
- 解説：https://zhuanlan.zhihu.com/p/117422157
- Demo動画：https://www.bilibili.com/video/av56708333

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- 論文：https://arxiv.org/abs/2003.08936

- コード：https://github.com/mit-han-lab/gan-compression

**Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions**

- 論文：https://arxiv.org/abs/2003.01826 
- コード：https://github.com/cc-hpc-itwm/UpConv 

<a name="Re-ID"></a>

# Re-ID

**Pose-guided Visible Part Matching for Occluded Person ReID**

- 論文：https://arxiv.org/abs/2004.00230
- コード：https://github.com/hh23333/PVPM

**Weakly supervised discriminative feature learning with state information for person identification**

- 論文：https://arxiv.org/abs/2002.11939 
- コード：https://github.com/KovenYu/state-information 

<a name="3D-PointCloud"></a>

# 3D点群（分類/分割/レジストレーション等）

## 3D点群畳み込み

**Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds**

- 論文下载链接：https://arxiv.org/abs/2003.12971

- コード：https://github.com/raoyongming/PointGLR

**Grid-GCN for Fast and Scalable Point Cloud Learning**

- 論文：https://arxiv.org/abs/1912.02984

- コード：https://github.com/Xharlie/Grid-GCN

**FPConv: Learning Local Flattening for Point Convolution**

- 論文：https://arxiv.org/abs/2002.10701
- コード：https://github.com/lyqun/FPConv

## 3D点群分類

**PointAugment: an Auto-Augmentation Framework for Point Cloud Classification**

- 論文：https://arxiv.org/abs/2002.10876 
- コード（公開前）： https://github.com/liruihui/PointAugment/ 

## 3D点群セマンティックセグメンテーション

**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**

- 論文：https://arxiv.org/abs/1911.11236
- コード：https://github.com/QingyongHu/RandLA-Net

- 解説：https://zhuanlan.zhihu.com/p/105433460

**Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels**

- 論文：https://arxiv.org/abs/2004.0409

- コード：https://github.com/alex-xun-xu/WeakSupPointCloudSeg

**PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation**

- 論文：https://arxiv.org/abs/2003.14032
- コード：https://github.com/edwardzhou130/PolarSeg

**Learning to Segment 3D Point Clouds in 2D Image Space**

- 論文：https://arxiv.org/abs/2003.05593

- コード：https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space

## 3D点群インスタンスセグメンテーション

PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

- 論文：https://arxiv.org/abs/2004.01658
- コード：https://github.com/Jia-Research-Lab/PointGroup

## 3D点群レジストレーション

**D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features**

- 論文：https://arxiv.org/abs/2003.03164
- コード：https://github.com/XuyangBai/D3Feat

**RPM-Net: Robust Point Matching using Learned Features**

- 論文：https://arxiv.org/abs/2003.13479
- コード：https://github.com/yewzijian/RPMNet 

## 3D点群补全

**Cascaded Refinement Network for Point Cloud Completion**

- 論文：https://arxiv.org/abs/2004.03327
- コード：https://github.com/xiaogangw/cascaded-point-completion

<a name="Face"></a>

# 顔

## 顔識別

**CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition**

- 論文：https://arxiv.org/abs/2004.00288

- コード：https://github.com/HuangYG123/CurricularFace

**Learning Meta Face Recognition in Unseen Domains**

- 論文：https://arxiv.org/abs/2003.07733
- コード：https://github.com/cleardusk/MFR
- 解説：https://mp.weixin.qq.com/s/YZoEnjpnlvb90qSI3xdJqQ 

## 顔検出

## Face Anti-spoofing

**Searching Central Difference Convolutional Networks for Face Anti-Spoofing**

- 論文：https://arxiv.org/abs/2003.04092

- コード：https://github.com/ZitongYu/CDCN

## 顔表情識別

**Suppressing Uncertainties for Large-Scale Facial Expression Recognition**

- 論文：https://arxiv.org/abs/2002.10392 

- コード（公開前）：https://github.com/kaiwang960112/Self-Cure-Network 

## Face Rotation

**Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images**

- 論文：https://arxiv.org/abs/2003.08124
- コード：https://github.com/Hangz-nju-cuhk/Rotate-and-Render

## 顔3D重建

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**

- 論文：https://arxiv.org/abs/2003.13845
- データセット：https://github.com/lattas/AvatarMe

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**

- 論文：https://arxiv.org/abs/2003.13989
- コード：https://github.com/zhuhao-nju/facescape

<a name="Human-Pose-Estimation"></a>

# 人体姿勢推定(2D/3D)

## 2D人体姿勢推定

**HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation**

- 論文：https://arxiv.org/abs/1908.10357
- コード：https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

**The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation**

- 論文：https://arxiv.org/abs/1911.07524 
- コード：https://github.com/HuangJunJie2017/UDP-Pose
- 解説：https://zhuanlan.zhihu.com/p/92525039

**Distribution-Aware Coordinate Representation for Human Pose Estimation**

- ホームページ：https://ilovepose.github.io/coco/ 

- 論文：https://arxiv.org/abs/1910.06278 

- コード：https://github.com/ilovepose/DarkPose 

## 3D人体姿勢推定

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**

- 論文下载链接：https://arxiv.org/abs/2004.01166

- コード：https://github.com/Healthcare-Robotics/bodies-at-rest
- データセット：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**Self-Supervised 3D Human Pose Estimation via Part Guided Novel Image Synthesis**

- ホームページ：http://val.cds.iisc.ac.in/pgp-human/
- 論文：https://arxiv.org/abs/2004.04400

**Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation**

- 論文：https://arxiv.org/abs/2004.00329
- コード：https://github.com/fabbrimatteo/LoCO

**VIBE: Video Inference for Human Body Pose and Shape Estimation**

- 論文：https://arxiv.org/abs/1912.05656 
- コード：https://github.com/mkocabas/VIBE

**Back to the Future: Joint Aware Temporal Deep Learning 3D Human Pose Estimation**

- 論文：https://arxiv.org/abs/2002.11251 
- コード：https://github.com/vnmr/JointVideoPose3D

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- 論文：https://arxiv.org/abs/2003.03972
- データセット：なし

<a name="Human-Parsing"></a>

# 人体解析

**Correlating Edge, Pose with Parsing**

- 論文：https://arxiv.org/abs/2005.01431

- コード：https://github.com/ziwei-zh/CorrPM

<a name="Scene-Text-Detection"></a>

# シーンテキスト検出

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 論文：https://arxiv.org/abs/2003.10608
- コードとデータセット：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- 論文：https://arxiv.org/abs/2002.10200 
- コード（公開前）：https://github.com/Yuliang-Liu/bezier_curve_text_spotting
- コード（公開前）：https://github.com/aim-uofa/adet

**Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection**

- 論文：https://arxiv.org/abs/2003.07493

- コード：https://github.com/GXYM/DRRG

<a name="Scene-Text-Recognition"></a>

# シーンテキスト識別

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 論文：https://arxiv.org/abs/2003.10608
- コードとデータセット：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- 論文：https://arxiv.org/abs/2002.10200 
- コード（公開前）：https://github.com/aim-uofa/adet

**Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition**

- 論文：https://arxiv.org/abs/2003.06606

- コード：https://github.com/Canjie-Luo/Text-Image-Augmentation

<a name="Super-Resolution"></a>

# 超解像

## 画像超解像

**Structure-Preserving Super Resolution with Gradient Guidance**

- 論文：https://arxiv.org/abs/2003.13081

- コード：https://github.com/Maclory/SPSR

**Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy**

論文：https://arxiv.org/abs/2004.00448

コード：https://github.com/clovaai/cutblur

## 動画超解像

**Space-Time-Aware Multi-Resolution Video Enhancement**

- ホームページ：https://alterzero.github.io/projects/STAR.html
- 論文：http://arxiv.org/abs/2003.13170
- コード：https://github.com/alterzero/STARnet

**Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution**

- 論文：https://arxiv.org/abs/2002.11616 
- コード：https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020 

<a name="Model-Compression"></a>

# モデル圧縮/枝刈り

**DMCP: Differentiable Markov Channel Pruning for Neural Networks**

- 論文：https://arxiv.org/abs/2005.03354
- コード：https://github.com/zx55/dmcp

**Forward and Backward Information Retention for Accurate Binary Neural Networks**

- 論文：https://arxiv.org/abs/1909.10788

- コード：https://github.com/htqin/IR-Net

**Towards Efficient Model Compression via Learned Global Ranking**

- 論文：https://arxiv.org/abs/1904.12368
- コード：https://github.com/cmu-enyac/LeGR

**HRank: Filter Pruning using High-Rank Feature Map**

- 論文：http://arxiv.org/abs/2002.10179
- コード：https://github.com/lmbxmu/HRank 

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- 論文：https://arxiv.org/abs/2003.08936

- コード：https://github.com/mit-han-lab/gan-compression

**Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression**

- 論文：https://arxiv.org/abs/2003.08935

- コード：https://github.com/ofsoundof/group_sparsity

<a name="Action-Recognition"></a>

# 動画理解/行動識別

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**

- ホームページ：https://sdolivia.github.io/FineGym/
- 論文：https://arxiv.org/abs/2004.06704

**TEA: Temporal Excitation and Aggregation for Action Recognition**

- 論文：https://arxiv.org/abs/2004.01398

- コード：https://github.com/Phoenix1327/tea-action-recognition

**X3D: Expanding Architectures for Efficient Video Recognition**

- 論文：https://arxiv.org/abs/2004.04730

- コード：https://github.com/facebookresearch/SlowFast

**Temporal Pyramid Network for Action Recognition**

- ホームページ：https://decisionforce.github.io/TPN

- 論文：https://arxiv.org/abs/2004.03548 
- コード：https://github.com/decisionforce/TPN 

## 基于骨架的动作識別

**Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition**

- 論文：https://arxiv.org/abs/2003.14111
- コード：https://github.com/kenziyuliu/ms-g3d

<a name="Crowd-Counting"></a>

# 群衆カウント

<a name="Depth-Estimation"></a>

# 深度推定

**AANet: Adaptive Aggregation Network for Efficient Stereo Matching**

- 論文：https://arxiv.org/abs/2004.09548
- コード：https://github.com/haofeixu/aanet

**Towards Better Generalization: Joint Depth-Pose Learning without PoseNet**

- 論文：https://github.com/B1ueber2y/TrianFlow

- コード：https://github.com/B1ueber2y/TrianFlow

## 単眼深度推定

**3D Packing for Self-Supervised Monocular Depth Estimation**

- 論文：https://arxiv.org/abs/1905.02693
- コード：https://arxiv.org/abs/1905.02693
- Demo動画：https://www.bilibili.com/video/av70562892/

**Domain Decluttering: Simplifying Images to Mitigate Synthetic-Real Domain Shift and Improve Depth Estimation**

- 論文：https://arxiv.org/abs/2002.12114
- コード：https://github.com/yzhao520/ARC

<a name="6DOF"></a>

# 6D物体姿勢推定

**EPOS: Estimating 6D Pose of Objects with Symmetries**

ホームページ：http://cmp.felk.cvut.cz/epos

論文：https://arxiv.org/abs/2004.00605

**G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features**

- 論文：https://arxiv.org/abs/2003.11089

- コード：https://github.com/DC1991/G2L_Net

<a name="Hand-Pose"></a>

# 手姿勢推定

**HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation**

- 論文：https://arxiv.org/abs/2004.00060

- ホームページ：http://vision.sice.indiana.edu/projects/hopenet

**Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data**

- 論文：https://arxiv.org/abs/2003.09572

- コード：https://github.com/CalciferZh/minimal-hand

<a name="Saliency"></a>

# 显著性検出

**JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection**

- 論文：https://arxiv.org/abs/2004.08515

- コード：https://github.com/kerenfu/JLDCF/

**UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders**

- ホームページ：http://dpfan.net/d3netbenchmark/

- 論文：https://arxiv.org/abs/2004.05763
- コード：https://github.com/JingZhang617/UCNet

<a name="Denoising"></a>

# 画像修復

**A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising**

- 論文：https://arxiv.org/abs/2003.12751

- コード：https://github.com/Vandermode/NoiseModel

**CycleISP: Real Image Restoration via Improved Data Synthesis**

- 論文：https://arxiv.org/abs/2003.07761

- コード：https://github.com/swz30/CycleISP

<a name="Deraining"></a>

# 去雨

**Multi-Scale Progressive Fusion Network for Single Image Deraining**

- 論文：https://arxiv.org/abs/2003.10985

- コード：https://github.com/kuihua/MSPFN

<a name="Deblurring"></a>

# デブラー

## 動画デブラー

**Cascaded Deep Video Deblurring Using Temporal Sharpness Prior**

- ホームページ：https://csbhr.github.io/projects/cdvd-tsp/index.html 
- 論文：https://arxiv.org/abs/2004.02501 
- コード：https://github.com/csbhr/CDVD-TSP

<a name="Dehazing"></a>

# 去雾

**Multi-Scale Boosted Dehazing Network with Dense Feature Fusion**

- 論文：https://arxiv.org/abs/2004.13388

- コード：https://github.com/BookerDeWitt/MSBDN-DFF

<a name="Feature"></a>

# 特徴点検出・記述

**ASLFeat: Learning Local Features of Accurate Shape and Localization**

- 論文：https://arxiv.org/abs/2003.10071

- コード：https://github.com/lzx551402/aslfeat

<a name="VQA"></a>

# VQA(VQA)

**VC R-CNN：Visual Commonsense R-CNN** 

- 論文：https://arxiv.org/abs/2002.12204
- コード：https://github.com/Wangt-CN/VC-R-CNN

<a name="VideoQA"></a>

# 動画问答(VideoQA)

**Hierarchical Conditional Relation Networks for Video Question Answering**

- 論文：https://arxiv.org/abs/2002.10698
- コード：https://github.com/thaolmk54/hcrn-videoqa

<a name="VLN"></a>

# 視覚言語ナビゲーション

**Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training**

- 論文：https://arxiv.org/abs/2002.10638
- コード（公開前）：https://github.com/weituo12321/PREVALENT

<a name="Video-Compression"></a>

# 動画圧縮

**Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement**

- 論文：https://arxiv.org/abs/2003.01966 
- コード：https://github.com/RenYang-home/HLVC

<a name="Video-Frame-Interpolation"></a>

# 動画補間

**Space-Time-Aware Multi-Resolution Video Enhancement**

- ホームページ：https://alterzero.github.io/projects/STAR.html
- 論文：http://arxiv.org/abs/2003.13170
- コード：https://github.com/alterzero/STARnet

**Scene-Adaptive Video Frame Interpolation via Meta-Learning**

- 論文：https://arxiv.org/abs/2004.00779
- コード：https://github.com/myungsub/meta-interpolation

**Softmax Splatting for Video Frame Interpolation**

- ホームページ：http://sniklaus.com/papers/softsplat
- 論文：https://arxiv.org/abs/2003.05534
- コード：https://github.com/sniklaus/softmax-splatting

<a name="Style-Transfer"></a>

# スタイル変換

**Diversified Arbitrary Style Transfer via Deep Feature Perturbation**

- 論文：https://arxiv.org/abs/1909.08223
- コード：https://github.com/EndyWon/Deep-Feature-Perturbation

**Collaborative Distillation for Ultra-Resolution Universal Style Transfer**

- 論文：https://arxiv.org/abs/2003.08436

- コード：https://github.com/mingsun-tse/collaborative-distillation

<a name="Lane-Detection"></a>

# 车道线検出

**Inter-Region Affinity Distillation for Road Marking Segmentation**

- 論文：https://arxiv.org/abs/2004.05304
- コード：https://github.com/cardwing/Codes-for-IntRA-KD

<a name="HOI"></a>

# Human-Object Interaction (HOT)検出

**Detailed 2D-3D Joint Representation for Human-Object Interaction**

- 論文：https://arxiv.org/abs/2004.08154

- コード：https://github.com/DirtyHarryLYL/DJ-RN

**Cascaded Human-Object Interaction Recognition**

- 論文：https://arxiv.org/abs/2003.04262

- コード：https://github.com/tfzhou/C-HOI

**VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions**

- 論文：https://arxiv.org/abs/2003.05541
- コード：https://github.com/ASMIftekhar/VSGNet

<a name="HTP"></a>

# 歩行者軌跡予測

**Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction**

- 論文：https://arxiv.org/abs/2002.11927 
- コード：https://github.com/abduallahmohamed/Social-STGCNN 

<a name="Motion-Predication"></a>

# モーション予測

**Collaborative Motion Prediction via Neural Motion Message Passing**

- 論文：https://arxiv.org/abs/2003.06594
- コード：https://github.com/PhyllisH/NMMP

**MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps**

- 論文：https://arxiv.org/abs/2003.06754

- コード：https://github.com/pxiangwu/MotionNet

<a name="Virtual-Try-On"></a>

# 虚拟试衣

**Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content**

- 論文：https://arxiv.org/abs/2003.05863
- コード：https://github.com/switchablenorms/DeepFashion_Try_On

<a name="HDR"></a>

# HDR

**Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline**

- ホームページ：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR

- 論文下载链接：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR_/00942.pdf

- コード：https://github.com/alex04072000/SingleHDR

<a name="AE"></a>

# 对抗样本

**Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance**

- 論文：https://arxiv.org/abs/1911.02466
- コード：https://github.com/ZhengyuZhao/PerC-Adversarial 

<a name="Datasets"></a>

# データセット

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**

- 論文：http://vladlen.info/papers/MSeg.pdf
- コード：https://github.com/mseg-dataset/mseg-api

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**

- 論文：https://arxiv.org/abs/2003.13845
- データセット：https://github.com/lattas/AvatarMe

**Learning to Autofocus**

- 論文：https://arxiv.org/abs/2004.12260
- データセット：なし

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**

- 論文：https://arxiv.org/abs/2003.13989
- コード：https://github.com/zhuhao-nju/facescape

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**

- 論文下载链接：https://arxiv.org/abs/2004.01166

- コード：https://github.com/Healthcare-Robotics/bodies-at-rest
- データセット：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**

- ホームページ：https://sdolivia.github.io/FineGym/
- 論文：https://arxiv.org/abs/2004.06704

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**

- ホームページ：https://anyirao.com/projects/SceneSeg.html

- 論文下载链接：https://arxiv.org/abs/2004.02678

- コード：https://github.com/AnyiRao/SceneSeg

**Deep Homography Estimation for Dynamic Scenes**

- 論文：https://arxiv.org/abs/2004.02132

- データセット：https://github.com/lcmhoang/hmg-dynamics

**Assessing Image Quality Issues for Real-World Problems**

- ホームページ：https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- 論文：https://arxiv.org/abs/2003.12511

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 論文：https://arxiv.org/abs/2003.10608
- コードとデータセット：https://github.com/Jyouhou/UnrealText/

**PANDA: A Gigapixel-level Human-centric Video Dataset**

- 論文：https://arxiv.org/abs/2003.04852

- データセット：http://www.panda-dataset.com/

**IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning**

- 論文：https://arxiv.org/abs/2003.02920
- データセット：https://github.com/intra3d2019/IntrA

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- 論文：https://arxiv.org/abs/2003.03972
- データセット：なし

<a name="Others"></a>

# その他

**Epipolar Transformers**

- 論文：なし

- コード：https://github.com/yihui-he/epipolar-transformers 

**Bringing Old Photos Back to Life**

- ホームページ：http://raywzy.com/Old_Photo/
- 論文：https://arxiv.org/abs/2004.09484

**MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask**

- 論文：https://arxiv.org/abs/2003.10955 

- コード：https://github.com/microsoft/MaskFlownet 

**Self-Supervised Viewpoint Learning from Image Collections**

- 論文：https://arxiv.org/abs/2004.01793
- 論文2：https://research.nvidia.com/sites/default/files/pubs/2020-03_Self-Supervised-Viewpoint-Learning/SSV-CVPR2020.pdf 
- コード：https://github.com/NVlabs/SSV 

**Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations**

- Oral

- 論文：https://arxiv.org/abs/2003.12237 
- コード：https://github.com/cuishuhao/BNM 

**Towards Learning Structure via Consensus for Face Segmentation and Parsing**

- 論文：https://arxiv.org/abs/1911.00957
- コード：https://github.com/isi-vista/structure_via_consensus

**Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging**

- Oral
- 論文：https://arxiv.org/abs/2003.13654

- コード：https://github.com/liuyang12/PnP-SCI

**Lightweight Photometric Stereo for Facial Details Recovery**

- 論文：https://arxiv.org/abs/2003.12307
- コード：https://github.com/Juyong/FacePSNet

**Footprints and Free Space from a Single Color Image**

- 論文：https://arxiv.org/abs/2004.06376

- コード：https://github.com/nianticlabs/footprints

**Self-Supervised Monocular Scene Flow Estimation**

- 論文：https://arxiv.org/abs/2004.04143
- コード：https://github.com/visinf/self-mono-sf

**Quasi-Newton Solver for Robust Non-Rigid Registration**

- 論文：https://arxiv.org/abs/2004.04322
- コード：https://github.com/Juyong/Fast_RNRR

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**

- ホームページ：https://anyirao.com/projects/SceneSeg.html

- 論文下载链接：https://arxiv.org/abs/2004.02678

- コード：https://github.com/AnyiRao/SceneSeg

**DeepFLASH: An Efficient Network for Learning-based Medical Image Registration**

- 論文：https://arxiv.org/abs/2004.02097

- コード：https://github.com/jw4hv/deepflash

**Self-Supervised Scene De-occlusion**

- ホームページ：https://xiaohangzhan.github.io/projects/deocclusion/
- 論文：https://arxiv.org/abs/2004.02788
- コード：https://github.com/XiaohangZhan/deocclusion

**Polarized Reflection Removal with Perfect Alignment in the Wild** 

- ホームページ：https://leichenyang.weebly.com/project-polarized.html
- コード：https://github.com/ChenyangLEI/CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment 

**Background Matting: The World is Your Green Screen**

- 論文：https://arxiv.org/abs/2004.00626
- コード：http://github.com/senguptaumd/Background-Matting

**What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective**

- 論文：https://arxiv.org/abs/2003.11241

- コード：https://github.com/ZhangLi-CS/GCP_Optimization

**Look-into-Object: Self-supervised Structure Modeling for Object Recognition**

- 論文：なし
- コード：https://github.com/JDAI-CV/LIO 

 **Video Object Grounding using Semantic Roles in Language Description**

- 論文：https://arxiv.org/abs/2003.10606
- コード：https://github.com/TheShadow29/vognet-pytorch 

**Dynamic Hierarchical Mimicking Towards Consistent Optimization Objectives**

- 論文：https://arxiv.org/abs/2003.10739
- コード：https://github.com/d-li14/DHM 

**SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization**

- 論文：http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf
- コード：https://github.com/YueJiang-nj/CVPR2020-SDFDiff 

**On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location**

- 論文：https://arxiv.org/abs/2003.07064

- コード：https://github.com/oskyhn/CNNs-Without-Borders

**GhostNet: More Features from Cheap Operations**

- 論文：https://arxiv.org/abs/1911.11907

- コード：https://github.com/iamhankai/ghostnet

**AdderNet: Do We Really Need Multiplications in Deep Learning?** 

- 論文：https://arxiv.org/abs/1912.13200 
- コード：https://github.com/huawei-noah/AdderNet

**Deep Image Harmonization via Domain Verification** 

- 論文：https://arxiv.org/abs/1911.13239 
- コード：https://github.com/bcmi/Image_Harmonization_Datasets

**Blurry Video Frame Interpolation**

- 論文：https://arxiv.org/abs/2002.12259 
- コード：https://github.com/laomao0/BIN

**Extremely Dense Point Correspondences using a Learned Feature Descriptor**

- 論文：https://arxiv.org/abs/2003.00619 
- コード：https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch

**Filter Grafting for Deep Neural Networks**

- 論文：https://arxiv.org/abs/2001.05868
- コード：https://github.com/fxmeng/filter-grafting
- 論文解説：https://www.zhihu.com/question/372070853/answer/1041569335

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**

- 論文：https://arxiv.org/abs/2003.02824 
- コード：https://github.com/cmhungsteve/SSTDA

**Detecting Attended Visual Targets in Video**

- 論文：https://arxiv.org/abs/2003.02501 

- コード：https://github.com/ejcgt/attention-target-detection

**Deep Image Spatial Transformation for Person Image Generation**

- 論文：https://arxiv.org/abs/2003.00696 
- コード：https://github.com/RenYurui/Global-Flow-Local-Attention

 **Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications** 

- 論文：https://arxiv.org/abs/2003.01455
- コード：https://github.com/bbrattoli/ZeroShotVideoClassification

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/Anonymous20192020/Anonymous_CVPR5767

https://github.com/avirambh/ScopeFlow

https://github.com/csbhr/CDVD-TSP

https://github.com/ymcidence/TBH

https://github.com/yaoyao-liu/mnemonics

https://github.com/meder411/Tangent-Images

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

https://github.com/sjmoran/deep_local_parametric_filters

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/bermanmaxim/AOWS

https://github.com/dc3ea9f/look-into-object 

<a name="Not-Sure"></a>

# 採択されたか不明

**FADNet: A Fast and Accurate Network for Disparity Estimation**

- 論文：未公開
- コード：https://github.com/HKBU-HPML/FADNet

https://github.com/rFID-submit/RandomFID

https://github.com/JackSyu/AE-MSR

https://github.com/fastconvnets/cvpr2020

https://github.com/aimagelab/meshed-memory-transformer

https://github.com/TWSFar/CRGNet

https://github.com/CVPR-2020/CDARTS

https://github.com/anucvml/ddn-cvprw2020

https://github.com/dl-model-recommend/model-trust

https://github.com/apratimbhattacharyya18/CVPR-2020-Corr-Prior

https://github.com/onetcvpr/O-Net

https://github.com/502463708/Microcalcification_Detection

https://github.com/anonymous-for-review/cvpr-2020-deep-smoke-machine

https://github.com/anonymous-for-review/cvpr-2020-smoke-recognition-dataset

https://github.com/cvpr-nonrigid/dataset

https://github.com/theFool32/PPBA

https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition

## 注意事項

このリポジトリは [amusi/CVPR2020-Code](https://github.com/amusi/CVPR2020-Code)を日本語訳したものです．

元のリポジトリの内容及び日本語訳は正確でないことがありますのでご注意ください．


## Acknowledgments

Proofread by Jin Jiongxing
