# Detection of Attacks on the Liberal Basic Democratic Order

This subdirectory contains the data for the second subtask of the shared task on "Harmful Content Detection in Social Media" within GermEval 2026: the **fine-grained detection of various attacks on the liberal democratic basic order of the Federal Republic of Germany**.

## Data Annotation

The dataset contains all tweets for which the three to four annotators could reach a majority decision regarding the form of attack. Specifically, the annotators assigned the tweets to one of the following classes:

| Class | Description |
|---|---|
| `subversive` | A will is expressed to forcibly remove the existing government and overthrow it (e.g., through militant action, disruption of the power grid, etc.). |
| `agitation` | Agitative efforts are expressed. That includes the announcement of actions such as the dissemination of propaganda material of unconstitutional and terrorist organisations or the damaging of state symbols such as the flag of the Federal Republic of Germany. |
| `criticism` | Tweets in which legitimate criticism of the government, officials, government employees, authorities or parties was expressed. |
| `nothing` | Tweets in this category contain neither criticism nor an attack against the free democratic basic order. However, neutral or positive statements on government decisions can be expressed in the tweets. |

## Origin and Structure of the Data

The **training data** for GermEval 2026 has been expanded and now includes a total of 15,853 tweets. The dataset consists predominantly of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The training data is provided as a CSV file. An entry has the following format:

```
"id";"description";"dbo"
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"nothing"
```

The **test dataset** contains 3,165 tweets. The test data is also distributed as a CSV file, containing an ID and the tweet text:

```
"id";"description"
```

## Anonymization of Data

To anonymise the data, mentions in the dataset were replaced as follows:

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
| `dbo_trial.csv` | Sample of the training dataset (~1,000 tweets), available since the trial phase to familiarise yourself with the data. |
| `dbo_train_26.csv` | Complete training dataset comprising 15,853 tweets. |
| `dbo_test_26.csv` | Complete training dataset comprising 3,165 tweets. |
