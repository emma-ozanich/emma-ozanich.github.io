---
layout: page
title: "Environmental data from NOAA BigQuery"
categories:
  - earth science
header:
  teaser: assets/images/windspeed_hveravellier.jpg
importance: 2
img: assets/img/windspeed_hveravellier.jpg
---

A demo I created for ECE 228 "Machine Learning for Physical Applications." It demonstrates how to pull, clean, visualize and analyze public data from NOAA GSOD dataset using Google BigQuery.
I have recently updated the demo to show how to predict seasonal timeseries trends: first, find the frequency/frequencies of the seasonality (one over the period); second, use seasonal ARIMA model trained on past data.


{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/demo.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/demo.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}