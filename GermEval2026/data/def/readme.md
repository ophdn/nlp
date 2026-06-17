# Detection of Defamatory Offences

This subdirectory contains the data for the fourth subtask of the shared task on "Harmful Content Detection in Social Media" within GermEval 2026: **the binary detection of defamatory offences** (i.e., Sections 185–187 of the German Criminal Code (StGB)).

## Data Annotation

The dataset contains all tweets for which a majority of the three annotators decided whether or not a statement in a tweet constitutes a defamatory offence under the German Criminal Code (StGB). Specifically, it must be checked whether any of the offences under Sections 185 to 187 of the German Criminal Code (StGB) have been committed. For information on the annotation of the dataset and the elements of the offence, please refer to the work of [Zufall et al. (2019)](https://aclanthology.org/N19-1135/).

## Origin and Structure of the Data

The **training data** for this pilot subtask contains 3,263 tweets. The main source of the dataset consists of posts and comments from a right-wing extremist movement from 12/12/2014 to 07/07/2016. The training data is provided as a CSV file. An entry has the following format:

```
"id";"description";"DEF"
"1064396393598783";"Oliver, ich guck doch schon mindestens einmal die Woche RTL2-NEWS.";"FALSE"
```

The **test data** contains 577 tweets and is also distributed as a CSV file, containing an ID and the tweet text:

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
| `def_trial.csv` | Sample of the training dataset (~100 tweets), available since the trial phase to familiarise yourself with the data. |
| `def_train.csv` | Complete training dataset comprising 3,263 tweets. |
| `def_test.csv` | Complete training dataset comprising 577 tweets. |
