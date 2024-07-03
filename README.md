# AutoEncoder Method in Anomaly Detection for SHM

## Introduction
![AD with AE model pic](https://github.com/happyleeyi/AnomalyDetection-with-AE-for-SHM/assets/173021832/5cac3b27-62ae-41fb-9a6d-96da29c50087)

The research to suggest a method to find whether and where the defect happens in the building on real-time.

We use an autoencoder method in anomaly detection to find the location of the defect in this section.

## Dataset

3-story shake table test data with an undamaged case and several damaged cases

![image](https://github.com/happyleeyi/SVDD-for-SHM/assets/173021832/b328aac8-2b3d-4063-9154-6c3e3cb029e1)

(open-source by Engineering Institute at Los Alamos National Laboratory)

## Method

### 1. data processing
1. prepare dataset (we use 3F shake table test dataset, 8 accelerometers per floor, a total of 24 accelerometers)
2. process dataset (we downsample and cut the training dataset and concatenate 8 acc data per each floor -> training dataset : (8, 512) for one data)

### 2. Training
![Training pic](https://github.com/happyleeyi/AnomalyDetection-with-AE-for-SHM/assets/173021832/7d93cc79-14c3-4185-b076-d2af86ce741f)

1. training autoencoder
2. put the training data into the trained autoencoder and make the reconstructed data, then calculate the reconstructed error of normal data
3. calculate the threshold value of the reconstructed error to determine whether the data is normal or abnormal in the test section (we calculated the threshold value as (threshold_quantile * max value of the reconstructed error of normal data))

### 3. Test
![Training pic - 복사본](https://github.com/happyleeyi/AnomalyDetection-with-AE-for-SHM/assets/173021832/b29f7841-d99d-4e31-8fd1-e902a8a79e56)

1. cut the test data for 3 floors (test dataset : (24, 512) -> (8,512) * 3 for one data)
2. put the test data into the autoencoder and calculate the reconstructed error for every floor
3. determine whether the data of each floor is normal or abnormal with a threshold value which is calculated in the training section
4. if every data is determined as normal, the building is on normal state
5. however, if several datas are determined as abnormal, the floor that the data's reconstructed error is the biggest is determined to be damaged.

## Result
![Distribution of the Reconstruction Loss](https://github.com/happyleeyi/AnomalyDetection-with-AE-for-SHM/assets/173021832/c5446e0a-e4d7-4b10-848f-522db21241ba)
![image](https://github.com/happyleeyi/AnomalyDetection-with-AE-for-SHM/assets/173021832/745c5a2f-af08-4122-9e6b-2fea065d6b5a)
![image](https://github.com/happyleeyi/AnomalyDetection-with-AE-for-SHM/assets/173021832/2dc4e637-d7cf-438b-a9ae-de1f391c189f)



