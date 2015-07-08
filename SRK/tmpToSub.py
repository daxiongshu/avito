import pandas as pd
import numpy as np

if __name__ == "__main__":
	sample = pd.read_csv('../Data/sampleSubmission.csv')
        preds = np.array(pd.read_csv('temp.csv', header = None))
        index = sample.ID.values - 1

        sample['IsClick'] = preds[index]
        sample.to_csv('submission9.csv', index=False)
