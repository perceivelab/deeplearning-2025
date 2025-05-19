---
layout: page
---

# Deep Learning Projects

---

## Project 1: Chest Radiograph Pathology Classification
- **Reference person:** Raffaele Mineo (raffaele.mineo@unict.it)
- **Goal:** To develop a model capable of predicting multiple pathologies from chest radiographs.
- **Dataset:**
    - Name: CheXpert
    - Description: 224,316 chest radiographs of 65,240 patients labeled with 14 observations (e.g., Cardiomegaly, Edema).
    - Link: [CheXpert on Kaggle](https://www.kaggle.com/datasets/ashery/chexpert)
    - Size: 11.5 GB
- **Methodology:** Train a convolutional neural network. Key challenges include handling uncertain labels, potentially using techniques like label smoothing or incorporating hierarchical dependencies between pathologies.
- **Performance evaluation:** Evaluate multi-label classification performance (e.g., AUC per label, F1-score).
- **Reference papers:**
    - Irvin, Jeremy, et al. "Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison." Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.
- **Reference/starting code:** [CheXpert Kaggle dataset and code](https://www.kaggle.com/datasets/ashery/chexpert/code)
- **Expected completion time:** 40h

---

## Project 2: Brain Tumor Segmentation
- **Reference person:** Raffaele Mineo (raffaele.mineo@unict.it)
- **Goal:** To segment different sub-regions of brain tumors (enhancing tumor, necrosis, and edema) from multi-parametric MRI scans.
- **Dataset:**
    - Name: BraTS-2021 (similar to BraTS2020)
    - Description: Multi-parametric MRI (T1, T1c, T2, FLAIR) of 2,040 glioma patients with voxel-wise labels.
    - Link: [BraTS2020 Training Data on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) (Note: Link is for BraTS2020, project mentions BraTS-2021)
    - Size: 8 GB
- **Methodology:** Implement a 3D U-Net architecture (or variants like nnU-Net) for volumetric segmentation.
- **Performance evaluation:** Evaluate segmentation accuracy using Dice scores on validation data for each sub-region.
- **Reference papers:**
    - Baid, Ujjwal, et al. "The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification." arXiv preprint arXiv:2107.02314 (2021).
- **Reference/starting code:** [BraTS2020 Kaggle dataset and code](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/code)
- **Expected completion time:** 60h

---

## Project 3: Diabetic Retinopathy Detection
- **Reference person:** Raffaele Mineo (raffaele.mineo@unict.it)
- **Goal:** To classify fundus images into five severity levels of diabetic retinopathy.
- **Dataset:**
    - Name: APTOS-2019 Blindness Detection
    - Description: 3,662 fundus images labeled into five DR severity levels (No DR, Mild, Moderate, Severe, Proliferative).
    - Link: [APTOS-2019 on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)
    - Size: 8.6 GB
- **Methodology:** Fine-tune a pre-trained Convolutional Neural Network (e.g., EfficientNet). Apply data augmentation techniques and address class imbalance using methods like SMOTE.
- **Performance evaluation:** Evaluate 5-class classification performance (e.g., Quadratic Weighted Kappa, accuracy, precision/recall per class).
- **Reference papers:**
    - Sikder, Niloy, et al. "Early blindness detection based on retinal images using ensemble learning." 2019 22nd International conference on computer and information technology (ICCIT). IEEE, 2019.
- **Reference/starting code:** [APTOS-2019 Kaggle dataset and code](https://www.kaggle.com/datasets/mariaherrerot/aptos2019/code)
- **Expected completion time:** 40h

---

## Project 4: Skin Lesion Classification
- **Reference person:** Raffaele Mineo (raffaele.mineo@unict.it)
- **Goal:** To classify dermoscopic images of skin lesions as benign or malignant.
- **Dataset:**
    - Name: ISIC-2020 (similar to ISIC-2019)
    - Description: 33,126 dermoscopic images (benign vs. malignant) confirmed via histopathology.
    - Link: [ISIC-2019 Skin Lesion Images on Kaggle](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) (Note: Link is for ISIC-2019, project mentions ISIC-2020)
    - Size: 9.8 GB
- **Methodology:** Build a classification pipeline, potentially using a ResNet backbone. Optionally, incorporate segmentation masks for lesion cropping prior to classification.
- **Performance evaluation:** Optimize and evaluate using ROC-AUC.
- **Reference papers:**
    - Cassidy, Bill, et al. "Analysis of the ISIC image datasets: Usage, benchmarks and recommendations." Medical image analysis 75 (2022): 102305.
- **Reference/starting code:** [ISIC-2019 Kaggle dataset and code](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification/code)
- **Expected completion time:** 50h

---

## Project 5: ECG Arrhythmia Classification
- **Reference person:** Raffaele Mineo (raffaele.mineo@unict.it)
- **Goal:** To classify 12-lead ECG signals for multi-label arrhythmia detection.
- **Dataset:**
    - Name: PTB-XL
    - Description: 21,799 clinical 12-lead ECGs (10s each) from 18,869 patients, annotated with up to 71 diagnostic statements (rhythm, form, diagnostic).
    - Link: [PTB-XL Dataset on Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset)
    - Size: 3.2 GB
- **Methodology:** Design a 1D-CNN or a Transformer-based model suitable for sequential ECG data.
- **Performance evaluation:** Compare performance across different diagnostic tasks (e.g., specific arrhythmia detection) using appropriate multi-label classification metrics.
- **Reference papers:**
    - Wagner, Patrick, et al. "PTB-XL, a large publicly available electrocardiography dataset." Scientific data 7.1 (2020): 1-15.
- **Reference/starting code:** [PTB-XL Kaggle dataset and code](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/code)
- **Expected completion time:** 40h

---

## Project 6: Histopathology Metastasis Detection
- **Reference person:** Raffaele Mineo (raffaele.mineo@unict.it)
- **Goal:** To classify patches from lymph node histology images for the presence of metastatic tissue.
- **Dataset:**
    - Name: PatchCamelyon (PCam)
    - Description: 327,680 color image patches (96x96px) from lymph node histology, with binary labels for metastatic tissue.
    - Link: [PatchCamelyon on Kaggle](https://www.kaggle.com/datasets/andrewmvd/metastatic-tissue-classification-patchcamelyon)
    - Size: 11.5 GB
- **Methodology:** Train a CNN (e.g., DenseNet). Experiment with semi-supervised learning techniques or uncertainty estimation to improve detection robustness.
- **Performance evaluation:** Evaluate classification performance (e.g., AUC, accuracy) for patch-based metastasis detection.
- **Reference papers:**
    - Veeling, Bastiaan S., et al. "Rotation equivariant CNNs for digital pathology." Medical image computing and computer assisted intervention–mICCAI 2018: 21st international conference, granada, Spain, September 16-20, 2018, proceedings, part II 11. Springer International Publishing, 2018.
- **Reference/starting code:** [Kaggle Histopathological Cancer Detection Challenge GitHub](https://github.com/ucalyptus/Kaggle-Histopathological-Cancer-Detection-Challenge)
- **Expected completion time:** 40h

---

## Project 7: Federated Adaptive Class-specific Prompt Tuning
- **Reference person:** Alessio Masano (alessio.masano@phd.unict.it)
- **Goal:** To adapt an existing federated multi-domain learning framework to support federated class-specific prompt tuning.
- **Dataset:**
    - Name: Food-101, FGVC Aircraft, Oxford Pets (one to be chosen, or preferably use all three)
    - Link: [Food-101](https://www.kaggle.com/datasets/dansbecker/food-101), [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [Oxford Pets](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset?select=images)
    - Size: Food-101 5GB, FGVC Aircraft 2.75GB, Oxford Pets 800MB 
- **Methodology:** Modify and extend an existing federated learning framework to incorporate class-specific prompt tuning mechanisms for adapting models across different domains/clients.
- **Performance evaluation:** Evaluate the effectiveness of class-specific prompt tuning in a federated setting, likely focusing on model performance on diverse data distributions and personalization.
- **Reference papers:**
    - Shangchao Su, Mingzhao Yang, Bin Li, Xiangyang Xue. "Federated Adaptive Prompt Tuning for Multi-Domain Collaborative Learning."
- **Reference/starting code:** [FedAPT GitHub](https://github.com/leondada/FedAPT)
- **Expected completion time:** 60h

---

## Project 8: Federated Adaptive Prompt Tuning in medical field
- **Reference person:** Alessio Masano (alessio.masano@phd.unict.it)
- **Goal:** To adapt and apply an existing federated multi-domain framework (likely FedAPT) using medical datasets.
- **Dataset:**
    - Name: MedMNIST v2 and MedIMeta (using domains with smaller size, e.g., Breast Ultrasound, Diabetic Retinopathy, Mammography, Skin Lesion Evaluation, etc.)
    - Link: [MedMNIST v2](https://medmnist.com/), [MedIMeta Zenodo](https://zenodo.org/records/7884735), [MedIMeta GitHub](https://github.com/StefanoWoerner/medimeta-pytorch/tree/master)
    - Size: Varies depending on selected subsets.
- **Methodology:** Apply the federated adaptive prompt tuning framework to medical image datasets, focusing on challenges specific to medical data like privacy, heterogeneity, and domain shift.
- **Performance evaluation:** Assess the model's performance on various medical tasks, its adaptability to different medical domains/institutions, and the benefits of prompt tuning in this context.
- **Reference papers:**
    - Shangchao Su, Mingzhao Yang, Bin Li, Xiangyang Xue. "Federated Adaptive Prompt Tuning for Multi-Domain Collaborative Learning."
- **Reference/starting code:** [FedAPT GitHub](https://github.com/leondada/FedAPT) (as a base)
- **Expected completion time:** 50h

---

## Project 9: Object Counting in Aerial Images
- **Reference person:** Morteza Moradi (morteza.moradi@unict.it)
- **Goal:** To count objects (e.g., animals) in drone or satellite images.
- **Dataset:**
    - Name: custom dataset of aerial images of animals
    - Link: https://datadryad.org/dataset/doi:10.5061/dryad.8931zcrv8
    - Size: 361.47 MB
- **Methodology:** Utilize YOLOv8 for initial object detection. Implement post-processing steps to refine counts, such as removing overlaps or applying clustering, potentially using segmentation-based refinement. Alternative approaches could involve density map estimation.
- **Performance evaluation:** Evaluate counting accuracy (e.g., Mean Absolute Error, RMSE) and detection performance if applicable.
- **Reference papers:**
    - (Implicit) Reference to models like "Counting animals in aerial images with a density map estimation model."
- **Reference/starting code:** YOLOv8 framework.
- **Expected completion time:** 60h

---

## Project 10: Lightweight Object Detection
- **Reference person:** Morteza Moradi (morteza.moradi@unict.it)
- **Goal:** To train a lightweight bounding box object detector for a custom dataset.
- **Dataset:**
    - Name: PASCAL VOC
    - Link: N/A (can be imported through PyTorch)
    - Size: 3 GB
- **Methodology:** Train an object detector using pretrained lightweight models such as MobileNet-SSD and/or YOLOv5n. Focus on adapting these models to a specific custom application.
- **Performance evaluation:** Evaluate object detection performance (e.g., mAP, speed) on the custom dataset.
- **Reference papers:**
    - "Performance Analysis of YOLOv3, YOLOv4 and MobileNet SSD for Real Time Object Detection."
- **Reference/starting code:** MobileNet-SSD and YOLOv5n implementations.
- **Expected completion time:** 50h

---

## Project 11: Medical Image Segmentation using Foundation Models
- **Reference person:** Federica Proietto Salanitri (federica.proiettosalanitri@unict.it)
- **Goal:** To identify a smaller, more efficient prompt encoder that, when combined with the frozen Segment Anything Model (SAM), achieves competitive segmentation performance on medical imaging tasks while reducing model size and computational cost.
- **Dataset:**
    - Name: NCI-ISBI2013
    - Link: [NCI-ISBI2013](https://tinyurl.com/tcia-prostate) 
    - Size: 613 MB
- **Methodology:** Adapt the Segment Anything Model (SAM) to medical images by exploring and integrating alternative, lightweight prompt encoders. The core SAM model is to remain frozen.
- **Performance evaluation:** Measure segmentation quality (e.g., Dice, IoU) and compare model size and computational cost against existing SAM adaptations like AutoSAM.
- **Reference papers:**
    - Shaharabany, Tal, et al. "AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder."
- **Reference/starting code:** [AutoSAM GitHub](https://github.com/talshaharabany/AutoSAM?tab=readme-ov-file)
- **Expected completion time:** 40h

---

## Project 12: Aligned Federated Learning with B-cos networks
- **Reference person:** Rutger Hendrix (rutger.hendrix@phd.unict.it)
- **Goal:** To explore weight-input alignment promoted by B-cos models as an implicit regularizer in federated learning to mitigate overfitting on local non-IID data.
- **Dataset:**
    - Name: Cifar10
    - Link: Standard dataset (e.g., via PyTorch torchvision)
    - Size: 170 MB
- **Methodology:** Start with pre-trained standard models distributed across clients. Apply B-cos transformations ("b-cosification") to these local models and then fine-tune them in a federated manner. An optional extension is to incorporate federated adaptive B (hyperparameter) tuning based on local needs (interpretability or performance).
- **Performance evaluation:** Assess model performance on non-IID data distributions, degree of overfitting, and potentially interpretability gains.
- **Reference papers:**
    - Dubost, Felix, et al. "B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable."
- **Reference/starting code:** implement [B-cosification](https://github.com/shrebox/B-cosification) transformation within a [federated learning](https://github.com/TsingZ0/PFLlib) framework.
- **Expected completion time:** 50h-60h

---

## Project 13: Uncertainty-aware adaptive learning in Segmentation
- **Reference person:** Rutger Hendrix (rutger.hendrix@phd.unict.it)
- **Goal:** To investigate uncertainty-aware learning techniques to improve model performance on uncertain regions of interest in image segmentation.
- **Dataset:**
    - Name: BraTs (Brain Tumor Segmentation dataset)
    - Link: Standard dataset (e.g., from MICCAI BraTS Challenge)
    - Size: N/A
- **Methodology:** Explore how targeted data augmentation strategies and/or specialized loss functions can be applied to areas where the model demonstrates high uncertainty during training. This aims to guide the model to dedicate more learning capacity to challenging areas.
- **Performance evaluation:** Evaluate segmentation performance, particularly in uncertain regions, and analyze the impact of uncertainty-guided techniques.
- **Reference papers:**
    - Yang, Xiaoguo, et al. "UGLS: an uncertainty guided deep learning strategy for accurate image segmentation."
- **Reference/starting code:** implementation of uncertainty estimation (e.g. [Evidential learning](https://github.com/Cocofeat/DEviS), [Monte Carlo Dropout](https://colab.research.google.com/drive/1m2Hy-K6CIw-S9TnTEbTCW0WCK9ivJn7O?usp=sharing),) and adaptive learning strategies (e.g data augmentation, adaptive loss).
- **Expected completion time:** 50h

---

## Project 14: Aligning human (eye-gaze) and model (saliency) attention
- **Reference person:** Rutger Hendrix  (rutger.hendrix@phd.unict.it)
- **Goal:** To align human attention (measured by eye-gaze) with model attention (through saliency maps).
- **Dataset:**
    - Name: MIMIC-GAZE-JPG
    - Link: https://github.com/ukaukaaaa/GazeGNN?tab=readme-ov-file
    - Size: 9 GB
- **Methodology:** Investigate different types of model saliency generation methods (e.g., layerwise relevance propagation, class activation mapping, guided backpropagation, occlusion methods), to identify those most suitable for training a classifier that is aligned with human saliency. This alignment is encouraged during training by incorporating an additional loss term that minimizes the difference between human and model saliency maps. 
- **Performance evaluation:** Measure the alignment/correlation between human eye-gaze data and model-generated saliency maps, and classification accuracy,  using appropriate metrics.
- **Reference papers:**
    - Boyd, Aidan, et al. "Cyborg: Blending human saliency into the loss improves deep learning-based synthetic face detection."
- **Reference/starting code:** N/A (requires implementation of various saliency methods and comparison framework).
- **Expected completion time:** 40h

---

## Project 15: Federated Subnetwork sharing
- **Reference person:** Rutger Hendrix (rutger.hendrix@phd.unict.it)
- **Goal:** To investigate 'information separation' in multi-input multi-output (MIMO) neural networks for federated learning, aiming to reduce communication costs by selectively sharing high-attribution subnetworks.
- **Dataset:**
    - Name: Cifar10
    - Link: Standard dataset (e.g., via PyTorch torchvision)
    - Size: 170 MB
- **Methodology:**
    1. Convert a standard CNN architecture to a MIMO configuration.
    2. Analyze how class-specific information flows through the network's bottleneck layers.
    3. Develop methods to identify high-attribution nodes (nodes crucial for specific classes).
    4. Demonstrate that selectively sharing only parameters related to these high-attribution nodes during federated learning can substantially reduce communication costs while maintaining model performance.
- **Performance evaluation:** Evaluate model classification accuracy, communication cost reduction, and the effectiveness of identifying and sharing class-attributing subnetworks.
- **Reference papers:**
    - Havasi, Marton, et al. "Training independent subnetworks for robust prediction."
- **Reference/starting code:** requires MIMO network implementation, identify attributing nodes, e.g. via [Taylor expention](https://github.com/VainF/Torch-Pruning/blob/master/tests/test_taylor_importance.py), and [federated learning](https://github.com/TsingZ0/PFLlib).
- **Expected completion time:** 60h

---

## Project 16: Identifying Brain Regions with fMRI
- **Reference person:** Marco Finocchiaro (finocchiaro.marco@phd.unict.it)
- **Goal:** To design a model capable of classifying the brain region to which a 3D voxel-based cube extracted from fMRI signals belongs.
- **Dataset:**
    - Name: Kamitani Lab dataset (referred to as "god-dataset" on Kaggle)
    - Description: fMRI data.
    - Link: [God Dataset on Kaggle](https://www.kaggle.com/datasets/marcofinocchiaro00/god-dataset)
    - Size: 7.5 GB
- **Methodology:** Develop a deep learning model (likely a 3D CNN) that takes 3D fMRI voxel cubes as input and classifies them into predefined brain regions.
- **Performance evaluation:** Assess classification accuracy for identifying brain regions.
- **Reference papers:** [Brain regions atlasing](https://arxiv.org/pdf/1612.03925)
- **Reference/starting code:** N/A
- **Expected completion time:** 40h
- **Can be extended to thesis**
---

## Project 17: Detecting Scalp Area from a Single EEG Electrode Signal
- **Reference person:** Marco Finocchiaro (finocchiaro.marco@phd.unict.it)
- **Goal:** To develop a data-driven pipeline capable of detecting specific scalp regions based on the signal captured by a single EEG electrode.
- **Dataset:**
    - Name: ORCA (EEG data from visual stimuli presentation)
    - Link: [Orca dataset on Kaggle](https://www.kaggle.com/datasets/marcofinocchiaro00/orca-dataset)
    - Size: 1.5 GB
- **Methodology:** Design a model (e.g., using signal processing and machine learning/deep learning techniques) to analyze single-electrode EEG signals and infer the corresponding scalp region.
- **Performance evaluation:** Evaluate the accuracy of scalp region detection.
- **Reference papers:** N/A
- **Reference/starting code:** N/A
- **Expected completion time:** 30-40h

---

## Project 18: Disentangling Color Semantics in CLIP: Discovering and Evaluating Sparse Sub-Spaces for Object Color Representation
- **Reference person:** Giovanni Bellitto (giovanni.bellitto@unict.it)
- **Goal:** To identify, characterize, and validate directions or sparse sub-spaces in CLIP’s latent space that predominantly encode the color attribute of an object.
- **Dataset:**
    - Name: N/A (will likely use CLIP models and probe them with various image/text inputs)
    - Link: N/A
    - Size: N/A
- **Methodology:**
    1. Assess if the transition from “base object” to “base object + color” is largely linear and consistent in CLIP's latent space.
    2. Determine if a small set of coordinates (or a low-density linear combination) explains most of the variance associated with color without sacrificing CLIP’s retrieval/classification performance.
    3. Verify if the same color-encoding direction is shared (or highly similar) across text and image modalities.
- **Performance evaluation:** Evaluate the linearity of color representation, the sparsity of the color sub-space, the impact on CLIP's core performance, and the alignment of color representation across modalities.
- **Reference papers:** N/A
- **Reference/starting code:** N/A (will involve working with pre-trained CLIP models).
- **Expected completion time:** 30h

---

## Project 19: Applying registers to B-Cos Networks
- **Reference person:** Amelia Sorrenti (amelia.sorrenti@phd.unict.it)
- **Goal:** To apply the concept of "registers" (from Vision Transformers) to B-cosified Vision Transformers (ViTs).
- **Dataset:**
    - Dataset choice might depend on the specific ViT and B-Cos models used, possibly ImageNet or similar. 
- **Methodology:** Integrate register tokens into the architecture of B-cosified Vision Transformers and evaluate their impact.
- **Performance evaluation:** Assess changes in model performance, interpretability, or other relevant metrics due to the introduction of registers in B-Cos ViTs.
- **Reference papers:**
    - "Vision Transformers Need Registers" [https://arxiv.org/abs/2309.16588]
    - "B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable" [https://arxiv.org/abs/2306.10898]
- **Reference/starting code:** 
    - ViT model with registers: https://github.com/kyegomez/Vit-RGTS
    - B-cos ViTs: https://github.com/B-cos/B-cos-v2
- **Expected completion time:** 30h
- **This project is indicated for students with personal computing resources or thesis interest.**

---

## Project 20: Continual Learning with registers
- **Reference person:** Amelia Sorrenti (amelia.sorrenti@phd.unict.it)
- **Goal:** To evaluate the impact of using "registers" (as in Vision Transformers) in a class-incremental continual learning setting.
- **Dataset:**
    - Standard datasets for evaluating robustness and continual learning.
    - Name: Imagenet-R (ImageNet-Renditions), CIFAR-100
    - Link:
        - Imagenet-R: https://github.com/hendrycks/imagenet-r
        - CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
    - Size:
        - Imagenet-R: 2 GB
        - CIFAR-100: 163 MB
- **Methodology:** Implement or adapt a continual learning setup (class-incremental) for a Vision Transformer model that incorporates registers.
- **Performance evaluation:** Assess continual learning metrics such as average accuracy, forgetting, and intransigence, comparing models with and without registers.
- **Reference papers:**
    - "Vision Transformers Need Registers" [https://arxiv.org/abs/2309.16588]
    - A Survey of Continual Learning: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10444954]
- **Reference/starting code:**
    - Requires setting up a continual learning experiment with ViTs and registers.
    - A practical starting point for working with continual learning: https://github.com/aimagelab/mammoth
        - This framework already includes dataloaders for the suggested datasets. 
    - ViT model with registers: https://github.com/kyegomez/Vit-RGTS
- **Expected completion time:** 30h
- **This project is indicated for students with personal computing resources or thesis interest.**

---

## Project 21: Dataset Distillation for CL
- **Reference person:** Matteo Pennisi (matteo.pennisi@unict.it)
- **Goal:** To create a buffer of distilled images using dataset distillation techniques and evaluate whether this improves continual learning (CL) performance.
- **Dataset:**
    - Name: CIFAR-10
- **Methodology:** Implement dataset distillation techniques to create a small, synthetic dataset (buffer) that summarizes a larger dataset. Use this distilled buffer in a continual learning scenario (e.g., as a memory replay mechanism).
- **Performance evaluation:** Evaluate standard continual learning metrics (accuracy, forgetting) when using the distilled dataset buffer compared to other CL strategies or no buffer.
- **Reference papers:**
    - [https://arxiv.org/abs/2203.11932](https://arxiv.org/abs/2203.11932)
- **Expected completion time:** 30h

---

## Project 22: Dataset Distillation for FL
- **Reference person:** Matteo Pennisi (matteo.pennisi@unict.it)
- **Goal:** To apply dataset distillation in a federated learning (FL) context, where each client creates a distilled version of its local dataset, and the server fine-tunes the global model on this aggregated distilled data.
- **Dataset:**
    - Name: Cifar-10
- **Methodology:**
    1. Each client in a federated network applies dataset distillation to its local data.
    2. Clients send their (small) distilled datasets to the server.
    3. The server aggregates these distilled datasets.
    4. The global federated model (e.g., obtained via FedAvg) is fine-tuned on the server using this aggregated distilled data before being shared back with clients.
- **Performance evaluation:** Evaluate FL performance (e.g., global model accuracy, communication efficiency) when using distilled datasets compared to standard FL approaches.
- **Reference papers:**
    - [https://arxiv.org/abs/2203.11932](https://arxiv.org/abs/2203.11932)
- **Expected completion time:** 30h

---

## Project 23: GenAI to explain CNN
- **Reference person:** Matteo Pennisi (matteo.pennisi@unict.it)
- **Goal:** To use generative AI (specifically Stable Diffusion) and lexical databases (Core WordNet) to explain the activations of a Convolutional Neural Network (CNN).
- **Dataset:**
    - Name: N/A (the "dataset" would be images generated by Stable Diffusion using Core WordNet words as prompts).
- **Methodology:**
    1. Generate images using Stable Diffusion with words from Core WordNet synsets as prompts.
    2. Pass these generated images through a pre-trained CNN.
    3. Analyze the CNN's activations to identify which WordNet synsets (and corresponding concepts/words) are most activated by the generated images, thereby providing an explanation for what the CNN "understands" or associates with those synsets.
- **Performance evaluation:** Qualitative and potentially quantitative assessment of how well the identified activated synsets explain the CNN's behavior or learned features.
- **Reference papers:**
    - (Implicit) Stable Diffusion: [https://huggingface.co/stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
    - (Implicit) Core WordNet: [https://wordnetcode.princeton.edu/standoff-files/core-wordnet.txt](https://wordnetcode.princeton.edu/standoff-files/core-wordnet.txt)
- **Reference/starting code:** N/A (requires using Stable Diffusion, WordNet, and analyzing CNN activations).
- **Expected completion time:** 30h

---

## Project 24: Static Images Network analyzer
- **Reference person:** Matteo Pennisi (matteo.pennisi@unict.it)
- **Goal:** To create a dataset specifically designed to detect common biases (e.g., color, background) in image classification models and use the analysis results to generate a report for each class, potentially using an LLM.
- **Dataset:**
    - Name: Custom-created dataset for bias detection.
    - Link: N/A
    - Size: N/A
- **Methodology:**
    1. Design and create a dataset where image attributes like object color, background, etc., are systematically varied while the main object class remains the same (or vice-versa) to probe for biases.
    2. Test a pre-trained image classification network on this dataset.
    3. Analyze the network's performance variations based on these controlled attributes to identify biases.
    4. Use an LLM to generate a summary report for each class, detailing detected biases.
- **Performance evaluation:** The "performance" is the quality and insightfulness of the bias detection and the generated reports.
- **Reference papers:** N/A
- **Reference/starting code:** N/A
- **Expected completion time:** 30h

---

## Project 25: Automatic check of explainability method using segmentation maps
- **Reference person:** Matteo Pennisi (matteo.pennisi@unict.it)
- **Goal:** To automatically evaluate the quality of an explainability method (Grad-CAM) by measuring the overlap between its generated saliency map and ground-truth segmentation masks or bounding boxes.
- **Dataset:**
    - Name: PASCAL VOC
    - Link: N/A (available from PyTorch)
    - Size: 3 GB
- **Methodology:**
    1. For a given image and a target class, generate an explainability map using Grad-CAM.
    2. Obtain a ground-truth segmentation mask or bounding box for the object of interest in the image.
    3. Quantitatively measure the overlap (e.g., using IoU or pointing game metrics) between the Grad-CAM saliency map and the ground truth.
- **Performance evaluation:** The metric is the degree of overlap, which serves as a proxy for the faithfulness/accuracy of the Grad-CAM explanation.
- **Reference papers:** N/A
- **Reference/starting code:** N/A (requires Grad-CAM implementation and evaluation pipeline).
- **Expected completion time:** 30h

---

## Project 26: Audio Image Alignement
- **Reference person:** Simone Carnemolla (simone.carnemolla@phd.unict.it)
- **Goal:** To align audio and image embeddings within a shared embedding space and perform zero-shot audio-image pair identification.
- **Dataset:**
    - Name: AVE (Audio-Visual Events dataset or similar)
    - Link: N/A (Standard dataset for audio-visual learning)
    - Size: N/A
- **Methodology:**
    - **Training:** Train a model (e.g., using two encoders for audio and image, and a contrastive loss) to map audio and image inputs into a common embedding space such that corresponding pairs are close and non-corresponding pairs are distant.
    - **Zero-shot Inference (Pair-wise Identification):**
        1. For each audio sample in the test set, select its corresponding image and C-1 randomly chosen distractor images from the test set (C is the number of classes).
        2. Calculate the embedding similarity (e.g., cosine similarity) between the audio sample's embedding and each of the C selected image embeddings.
        3. The model's prediction is the audio-image pair with the highest similarity score.
- **Performance evaluation:** Pair-wise identification accuracy: a prediction is correct if the highest similarity occurs between the audio-image pair that belongs to the same class.
- **Reference papers:**
    - (Implicit) CLIP-CLAP or similar audio-visual alignment papers.
- **Reference/starting code:** https://github.com/openai/CLIP
- **Expected completion time:** 30h-40h

---

## Project 27: Textual Backward
- **Reference person:** Simone Carnemolla (simone.carnemolla@phd.unict.it)
- **Goal:** To investigate the decision-making process of a visual classifier using a training-free framework that simulates a training process through textual instructions and iterative prompt refinement with an LLM.
- **Dataset:**
    - Name: Imagenet 1k (specifically, Salient Imagenet or a subset of 5 biased classes from ImageNet).
    - Link: Standard ImageNet dataset.
- **Methodology:**
    - **Setup:** Visual Classifier (e.g., RobustResNet50), LLM for prompt generation/refinement.
    - **Optimization Pipeline (per class):**
        1. LLM generates an initial textual prompt for a target class.
        2. Generate an image from this prompt (e.g., using a text-to-image model).
        3. Classify the generated image using the visual classifier.
        4. Provide the classifier's prediction, the target class, and the original prompt to the LLM to generate an improved prompt.
        5. Repeat for 'n' optimization steps.
    - **Inference Pipeline:** Use the final optimized prompt to generate 100 images.
- **Performance evaluation:** Measure how many of the 100 images generated using the final optimized prompt activate the target class in the visual classifier. Success is based on developing system prompts for initial/refined generation and implementing the pipeline. Optional: Integrate an LLM for textual loss or auxiliary agents.
- **Reference papers:**
    - (Implicit) TextGrad or similar papers on understanding models via generative approaches.
- **Reference/starting code:** N/A (requires interaction between LLMs, text-to-image models, and classifiers).
- **Expected completion time:** 30h-40h
- **Can be extended to thesis**

---

## Project 28: Segmentation using bbox from GroundingSAM
- **Reference person:** Giovanni Patanè (patane.giovanni@phd.unict.it)
- **Goal:** To leverage bounding boxes obtained from GroundingSAM (or a similar open-world object detector) to perform or refine image segmentation.
- **Dataset:**
    - Name: GOOSE 2D Images.
    - Link: [GOOSE DATASET](https://www.kaggle.com/datasets/mrgiop/goose-ex-2d-images) 
    - Size: 15 GB
- **Methodology:** Integrate GroundingSAM (or a model that provides open-world object detection with bounding boxes based on text prompts) with a segmentation model. The bounding boxes could be used as prompts or constraints for a segmentation model (like SAM itself, or another architecture) to achieve more precise or class-specific segmentation.
- **Performance evaluation:** Evaluate segmentation performance (e.g., IoU, Dice score) based on the input prompts/bounding boxes.
- **Reference papers:**
    - (Implicit) "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks."
    - (Dataset) [Excavating in the Wild: The GOOSE-Ex Dataset for Semantic Segmentation](https://goose-dataset.de/images/gooseEx.pdf)
- **Reference/starting code:**  [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Grounding SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [GOOSE Dataset: Image Processing](https://github.com/FraunhoferIOSB/goose_dataset/tree/main/image_processing) 
- **Expected completion time:** ~30-40h

---

## ~~Project 29: Classification Breast Cancer~~
- **Reference person:** Giovanni Patanè (patane.giovanni@phd.unict.it)
- **Goal:** To train or fine-tune a deep learning model to classify abnormal areas of tissue (benign vs. malignant) in mammography images.
- **Dataset:**
    - Name: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
    - Link: [Standard medical imaging dataset for mammography](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).
    - Size: 965 KB for Mass cases classification
- **Methodology:**
    1. Preprocess mammography images from the CBIS-DDSM dataset.
    2. Train or fine-tune a deep learning model.
- **Performance evaluation:** Evaluate model performance using appropriate metrics (e.g., AUC for classification, IoU/Dice for segmentation).
- **Reference papers:**
    - "Transfer Learning and Fine Tuning in Breast Mammogram Abnormalities Classification on CBIS-DDSM Database."
- **Expected completion time:** 30h-40h~~

---

## Project 30: Behaviour decoding from Mouse Visual Cortex
- **Reference person:** Salvatore Calcagno (salvatore.calcagno@unict.it)
- **Goal:** To determine if it's possible to decode behaviour using 2p brain signals.
- **Dataset:**
    - Name: SENSORIUM
    - Link: [https://github.com/sinzlab/sensorium?tab=readme-ov-file#sensorium-2022-competition](https://github.com/sinzlab/sensorium?tab=readme-ov-file#sensorium-2022-competition) (download only - no need for docker installation)
    - Size: 500 MB / mouse
- **Methodology:** Given 2p traces (from SENSORIUM or similar), train a model to predict mouse behavioural variables (pupil size and position, speed)
- **Performance evaluation:**
    - MSE between real and predicted behaviour
    - Correlation between real and predicted behaviour
- **Reference papers:** https://arxiv.org/abs/2206.08666
- **Expected completion time:** 30h

---

## Project 31: Image Classification from Brain Signals
- **Reference person:** Salvatore Calcagno (salvatore.calcagno@unict.it)
- **Goal:** To determine if it's possible to classify the category of seen objects (e.g., animate vs. inanimate, nature vs. artificial) using only brain signals.
- **Dataset:**
    - Name: SENSORIUM
    - Link: [https://github.com/sinzlab/sensorium?tab=readme-ov-file#sensorium-2022-competition](https://github.com/sinzlab/sensorium?tab=readme-ov-file#sensorium-2022-competition) (download only - no need for docker installation)
    - Size: 500 MB / mouse
- **Methodology:** Given brain representations of images (from SENSORIUM or similar), train a classifier to predict user-defined categories of the objects viewed by the subject. Images can also be used during the training phase to help learn the mapping or representations.
- **Performance evaluation:** Classification accuracy for the defined categories based on brain signals.
- **Reference papers:** https://arxiv.org/abs/2206.08666
- **Expected completion time:** 30h

---

## Project 32: Brain Signal Quantization
- **Reference person:** Salvatore Calcagno (salvatore.calcagno@unict.it)
- **Goal:** To evaluate an advanced quantization method against a VQ-VAE baseline on 2-photon ΔF/F (calcium imaging) traces.
- **Dataset:**
    - Name: Allen Brain Observatory (choose one session)
    - Link: Data: [Allen Visual Coding Observatory](https://observatory.brain-map.org/visualcoding), Tutorials: [Allen Observatory Examples GitHub](https://github.com/AllenInstitute/brain_observatory_examples/blob/master/Visual%20Coding%202P%20simple%20tutorial.ipynb)
    - Size: 100 MB
- **Methodology:**
    1. Implement a VQ-VAE with a 512-entry codebook as a baseline.
    2. Choose an advanced quantization method (e.g., from audio domain references) and replace the VQ-VAE's quantization layer with it, keeping the same encoder/decoder.
    3. Train both models under identical conditions on 2-photon ΔF/F traces.
    4. Evaluate by training a linear decoder on the quantized representations to predict behavioral variables (e.g., eye position, running speed) or stimulus information.
- **Performance evaluation:** Report metrics such as:
    - MSE on reconstruction of the ΔF/F traces.
    - Codebook entropy (% of used codes).
    - Performance of decoding behavioral or stimulus variables from the quantized representations.
    (Success is not strictly required, focus is on the investigation).
- **Reference papers:**
    - Quantization methods used in audio: [https://arxiv.org/pdf/2502.06490](https://arxiv.org/pdf/2502.06490)
- **Reference/starting code:** [https://github.com/SalvoCalcagno/quantformer2024](https://github.com/SalvoCalcagno/quantformer2024)
- **Expected completion time:** 40h+

---

## Project 33: ~~Few-shot Predictive Maintenance Benchmarking~~
- **Reference person:** Simone Palazzo (simone.palazzo@unict.it)
- **Goal:** To evaluate the effectiveness of different approaches in estimating the Remaining Useful Lifetime (RUL) of a power electronic device, given a few time series samples of a temperature sensor.
- **Dataset:**
    - Name: Private dataset
    - Description: 8 samples, ~400 timesteps per sample of temperature sensor data.
    - Size: ~1 MB
- **Methodology:**
    1. Evaluate CNN and Transformer solutions, trained from scratch on the few-shot data.
    2. Evaluate CNN and Transformer solutions, using models pretrained on other (larger, possibly related) time series datasets and then fine-tuned.
    3. Propose and evaluate data augmentation and/or preprocessing procedures suitable for time series in a few-shot context.
- **Performance evaluation:**
    - Remaining Useful Lifetime (RUL) plots.
    - Area Under the Curve (AUC) of RUL plot.
- **Reference papers:**
    - [https://ieeexplore.ieee.org/document/10315015](https://ieeexplore.ieee.org/document/10315015)
- **Reference/starting code:** Provided for dataset loading, a baseline LSTM model training, and plot generation.
- **Expected completion time:** 30h

