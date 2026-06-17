# Violence Detection

This subdirectory contains the data for the third subtask of the shared task on "Harmful Content Detection in Social Media" within GermEval 2026: the **fine-grained classification of disturbing statements about violence**.

## Data Annotation

The dataset contains all tweets for which the three to four annotators reached a majority decision regarding the type of violent statement. Specifically, a detailed annotation was made in five subtypes of violent statements and one negative category (six categories in total):

| Class | Description |
|---|---|
| `nothing` | No violent expression, i.e., no negative statements about violence at all. |
| `propensity` | Willingness to commit violence, i.e., the will or desire to use violence oneself. |
| `call2Violence` | Call to violence, i.e., inciting or calling on other people to commit a violent act. |
| `support` | Endorsement of violence, i.e., positive approval of violence or a violent event. |
| `glorification` | Glorification of violence, i.e., violence is presented as something particularly glorious and not just supported. |
| `other` | Other forms of worrying, violence-related statements. |

## Origin and Structure of the Data

The dataset consists predominantly of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. A total of 20,539 tweets were annotated for violence detection, divided into training and test data using stratified sampling at a ratio of 80:20.

The **training data** comprises 16,431 tweets and is provided as a CSV file. An entry has the following format:

```
"id";"description";"vio"
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"nothing"
```

The **test dataset** contains 4,108 tweets and is also distributed as a CSV file, containing an ID and the tweet text:

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
| `vio_trial.csv` | Sample of the training dataset (~1,000 tweets), available since the trial phase to familiarise yourself with the data. |
| `vio_train_26.csv` | Complete training dataset comprising 16,431 tweets. |
| `vio_test_26.csv` | Complete training dataset comprising 4,108 tweets. |
