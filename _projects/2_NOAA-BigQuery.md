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

I created this demo for ECE 228 "Machine Learning for Physical Applications." It demonstrates how to load, visualize, and analyze data from the NOAA GSOD dataset.
In 2023, I've added a demo for prediction of seasonal timeseries, using the FFT to find major seasonal cycles, and SARIMAX (seasonal arima model) to predict.


{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/demo.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/demo.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}