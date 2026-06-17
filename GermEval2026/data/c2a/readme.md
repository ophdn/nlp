# Call to Action

This subdirectory contains the data for the first subtask of the shared task on "Harmful Content Detection in Social Media" within GermEval 2026: **binary detection of calls to action**.

## Data Annotation

The dataset contains all tweets for which a majority of the three to four annotators decided whether a tweet was a call to action (`TRUE`) or not (`FALSE`). A call to action is understood to be, based on the definition of the [Oxford Dictionaries](https://www.oxfordlearnersdictionaries.com/definition/english/call-to-action), an order or request for a specific action or behaviour. The behaviour that a call to action encourages or incites may, but does not have to be, criminally relevant. For example, it may also be a call for a demonstration or political campaign, such as distributing leaflets.

## Origin and Structure of the Data

The **training data** for GermEval 2026 has been expanded to 15,829 tweets. The dataset consists of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The dataset is provided as a CSV file that includes the ID, text, and call-to-action label. Each entry has the following format:

```
"id";"description";"c2a";
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";FALSE;
```

The **test dataset** contains 2,982 tweets. It is identical to the GermEval 2025 test set to allow direct comparability between editions. The test data is also distributed as a CSV file, containing an ID and the tweet text:

```
"id";"description"
```

## Anonymization of Data

To anonymise the data, mentions in the dataset (training and test data) were replaced as follows:

| Placeholder | Replaced mention |
|---|---|
| `[@PRE]` | Mentions of the press / press offices / news portals |
| `[@POL]` | Mentions of the police / police authorities |
| `[@GRP]` | Mentions of groups / organisations / associations |
| `[@IND]` | Mentions of individuals |

**Example:** The mentions of the organisation Greenpeace and the TV channel ARD in the following (fictitious) tweet would be replaced as follows:

> *@greenpeace_de Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken @ARDde*
>
> ⟹ *[@GRP] Euch liegt bei euren Aktionen wohl etwas an Sicherheit. Da muss man sich ja nur die letzte Doku ansehen, um das zu merken [@PRE]*

No further preprocessing steps were performed on the data.

## Files

| File | Description |
|---|---|
| `c2a_trial.csv` | Sample of the training dataset (~1,000 tweets), available since the trial phase to familiarise yourself with the data. |
| `c2a_train_26.csv` | Complete training dataset comprising 15,829 tweets. |
| `c2a_test_26.csv` | Test dataset comprising 2,982 tweets. |
