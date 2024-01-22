---
layout: page
title: Introduction to Anomaly Detection
description: Anomaly Detection
img: 
importance: 4
category: fun
giscus_comments: true
---

## What is Anamoly Detection?

Anamoly detection (outlier detection), is the process of finding patterns or instances in a dataset that deviate significantly from the expected behavior.

A massive withdrawl from a location that user does not normally perform consistant spending, or an abrupt increase in data transfer or use of unknown protocols are all examples of an outlier detection.

## Types of Anomalies and outliers

There are two types of anomaly detection: outlier detection and novelty detection.

- Outliers are abnormal or extreme data points that exist only in training data. 
- Novelties are new or previously unseen instances compared to the original (training) data.

As there are two types of anomalies, there are two types of outliers as well: **univariate and multivariate**. Depending on the type, we will use different detection algorithms.

- Univariate outliers exist in a single variable or feature in isolation. Univariate outliers are extreme or abnormal values that deviate from the typical range of values for that specific feature.

- Multivariate outliers are found by combining the values of multiple variables at the same time.

## Anomaly detection methods
For univariate outlier detection, the most popular methods are:

1. Z-score (standard score): the z-score measures how many standard deviations a data point is away from the mean. Generally, instances with a z-score over 3 are chosen as outliers.

2. Interquartile range (IQR): The IQR is the range between the first quartile (Q1) and the third quartile (Q3) of a distribution. When an instance is beyond Q1 or Q3 for some multiplier of IQR, they are considered outliers. The most common multiplier is 1.5, making the outlier range [Q1–1.5 * IQR, Q3 + 1.5 * IQR].

3. Modified z-scores: similar to z-scores, but modified z-scores use the median and a measure called Median Absolute Deviation (MAD) to find outliers. Since mean and standard deviation are easily skewed by outliers, modified z-scores are generally considered more robust.

For multivariate outliers, we generally use machine learning algorithms. Because of their depth and strength, they are able to find intricate patterns in complex datasets:

1. Isolation Forest: uses a collection of isolation trees (similar to decision trees) that recursively divide complex datasets until each instance is isolated. The instances that get isolated the quickest are considered outliers.
2. Local Outlier Factor (LOF): LOF measures the local density deviation of a sample compared to its neighbors. Points with significantly lower density are chosen as outliers.
3. Clustering techniques: techniques such as k-means or hierarchical clustering divide the dataset into groups. Points that don’t belong to any group or are in their own little clusters are considered outliers.
4. Angle-based Outlier Detection (ABOD): ABOD measures the angles between individual points. Instances with odd angles can be considered outliers.
