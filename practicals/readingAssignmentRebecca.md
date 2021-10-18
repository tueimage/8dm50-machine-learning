# Discusses the strong and weak points of the methodology and evaluation metrics

## Weak points 
- They made certain all patients they labelled as "cancer" actually had cancer, but they say it is possible some patients labelled as "not cancer" may have developed cancer and may therefore be labelled wrongly, inducing a bias in the model leaning towards more false negatives, possibly reducing the sensitivity of the model.
- Only the 10 largest nodules are considered, an 11th cancerous nodule may be overlooked, further reducing sensitivity.
- The model does not tell which nodule was malignant, making intervention based on this model difficult as manual determination of all nodules is still required.

## Strong points 
- They made very certain not to have data overlap between training and testing datasets.


## Summary
They made a model that seems to have a very good performance, but the way the model is to be applied is not clear from the paper. The model may have a slight reduced sensitivity bias because of the afore mentioned weak points, so it may not be beneficial to treat only patients labelled as "cancer" as you may miss some. It is neither useful to treat all "cancer" labelled patients and still manually test the rest, because the model does not tell which module was malignant, so that way you are still manually testing all patients. 
