# 🔍 Harmful Content Detection
## A GermEval 2026 Shared Task

[![Register](https://img.shields.io/badge/Register-CodaBench-blue)](https://www.codabench.org/competitions/edit/14006/#/)
[![Phase](https://img.shields.io/badge/Current%20Phase-Training-green)](#take-part-in-the-shared-task)
[![Tweets](https://img.shields.io/badge/Annotated%20Tweets-22%2C200-orange)](#overview-of-the-data)

This repository contains the annotated Twitter dataset provided for the shared task in the context of GermEval 2026. The shared task is organised into the following subtasks:

| | Subtask | Description |
|---|---|---|
| 1 | 📢 **Calls to Action** | Binary detection of calls for risky actions (e.g., criminal offences, demonstrations with possible escalation potential) |
| 2 | ⚖️ **Liberal Democratic Basic Order** | Fine-grained classification into four forms of statements and (violent) attacks against the liberal democratic basic order of the Federal Republic of Germany |
| 3 | 🔥 **Violence-Related Statements** | Fine-grained classification into six different categories of violence-related statements in tweets |
| 4 | 🚨 **Defamatory Offences** | Binary detection of defamatory offences (i.e., Sections 185-187 of the German Criminal Code (StGB)) |

## Overview of the Data

For GermEval 2026, a sample of around **22,200 tweets** from a total corpus of approximately 800,000 tweets was annotated. The data sets for **subtasks 1–3** are based on data already annotated in the previous edition and were expanded for GermEval 2026.

The annotation of the first three subtasks was carried out by members of Mittweida University of Applied Sciences. Each tweet was annotated by three to four annotators. Only tweets for which there was a majority decision by the three to four annotators were included in the final data sets. This resulted in training datasets comprising approximately **15,500-16,500** tweets for the first three subtasks and approximately 3,000 tweets for the fourth subtask.

The particularly challenging annotation of the fourth sub-task was carried out by the Shared Task organizers, other staff members from Mittweida University of Applied Sciences, and the central authority for information technology in the security sector (ZITiS), all of whom have extensive practical experience in assessing harmful and criminally relevant content.

The data sets for each subtask, and further explanations of the data, can be found in the repository's individual subdirectories.

## Take Part in the Shared Task

To take part in this competition, please register [here](https://www.codabench.org/competitions/edit/14006/#/).

**Important Deadlines:**

| **Date** | | **Phase/Deadline** |
|---|---|---|
| ✅ 21 February - 25 March 2026 | | Trial phase |
| 18 April - 23 May 2026 | | Training phase ← current |
| 👉 **24 May - 21 June 2026** | | Competition phase |
| 15 July 2026 | | Paper submission due |
| 15 August 2026 | | Camera ready due |
