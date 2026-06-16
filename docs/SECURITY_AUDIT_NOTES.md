# Security Audit Notes

## Purpose
Preliminary repository audit before institutional Power BI/Azure handoff.

## Scope
Reviewed for:
- Hardcoded secrets
- API keys
- Credentials
- Sensitive Data Files
- Generated Outputs
- GitHub Actions workflows
- Envrionment variable usage
- Cloud deployment configuration
- Files that should not be tracked in GitHub

## Findings

Generated data artifacts were found in the repository. These files are created by the nightly pipeline and should not be stored in GitHub. They should be written to managed cloud storage, currently GCS, during the nightly run. The active Git branch will remove these files from source control and retain only code, configuration templates, documentation, and sanitized sample data.

| Area | Finding | Risk | Action Taken | Remaining Concern |
|---|---|---|---|---|
| Secrets | Pending Review | Credential Exposure | Pending | Pending |
| Data files | Pending Review | Sensitive data exposure | Pending | Pending |
| GitHub actions | Pending Review | CI/CD exposure | Pending | Pending |
| Cloud config | Pending Review | Over-permissioned access | Pending | Pending |
| Outputs/logs | Pending Review | Data leakage | Pending | Pending |


| File                                                    | Action                                       | Reason                                                                                                | Future Location                   |
| ------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| BERTopic_before.csv                                     | Deleted                                      | Duplicate artifact; no longer needed                                                                  | N/A                               |
| pipeline/resources/Articles_with_Stories.csv.gz         | Removed from Git tracking                    | Generated pipeline output; recreatable from nightly pipeline; not source code                         | GCS                               |
| pipeline/resources/BERTopic_before.csv                  | Removed from Git tracking                    | Generated BERTopic artifact; recreatable from pipeline                                                | GCS                               |
| pipeline/resources/BERTopic_changes.csv                 | Review before migration                      | Runtime state file used by Streamlit topic editing workflow; dependency must be verified              | GCS or Database                   |
| pipeline/resources/Canonical_Stories_with_Summaries.csv | Removed from Git tracking                    | Generated story summarization output; recreatable from pipeline                                       | GCS                               |
| pipeline/resources/Story_Clusters.csv.gz                | Removed from Git tracking                    | Generated clustering output; recreatable from pipeline                                                | GCS                               |
| pipeline/resources/dashboard_articles.csv.gz            | Removed from Git tracking                    | Generated dashboard output; not source code                                                           | GCS / Reporting Layer             |
| pipeline/resources/dashboard_dropdown.csv.gz            | Removed from Git tracking                    | Generated dashboard output; not source code                                                           | GCS / Reporting Layer             |
| pipeline/resources/dashboard_stories.csv.gz             | Removed from Git tracking                    | Generated dashboard output; not source code                                                           | GCS / Reporting Layer             |
| pipeline/resources/final_risk_scores1.csv               | Removed from Git tracking                    | Generated institutional risk intelligence output                                                      | Database / Reporting Layer        |
| pipeline/resources/la_legis_bills.json                  | Remove and replace with automated extraction | Static extracted dataset should not be manually maintained                                            | Automated extraction → GCS        |
| pipeline/resources/risk_mlb.pkl                         | Flagged for migration                        | Serialized ML artifact; required by runtime but should not remain in source control long-term         | Controlled Model Storage / GCS    |
| pipeline/resources/risk_predictions.csv                 | Removed from Git tracking                    | Generated model prediction output; recreatable from pipeline                                          | GCS / Database                    |
| pipeline/resources/risks.json | Removed from Git tracking | Contains institutional risk taxonomy and risk intelligence definitions; should not be publicly exposed through source control | Restricted GCS Storage / Future Azure Managed Storage |
| pipeline/resources/story_centroids.pkl                  | Removed from Git tracking                    | Generated clustering artifact; recreatable from training process                                      | GCS / Controlled Artifact Storage |
| pipeline/resources/subtopic_centroids.pkl               | Removed from Git tracking                    | Generated clustering artifact; recreatable from training process                                      | GCS / Controlled Artifact Storage |
| pipeline/resources/subtopics.csv                        | Removed from Git tracking                    | Generated topic modeling output; recreatable from pipeline                                            | GCS                               |
| pipeline/resources/test_articles.csv                    | Deleted                                      | Test data should not be stored in production repository                                               | N/A                               |
| pipeline/resources/topic_trend.csv                      | Deleted                                      | Generated trend output; recreatable from pipeline                                                     | GCS / Reporting Layer             |
| pipeline/resources/topics_BERT.json                     | Removed from Git tracking                    | Generated BERTopic metadata artifact                                                                  | GCS / Controlled Artifact Storage |
| pipeline/resources/train_articles.csv                   | Deleted                                      | Training dataset should not reside in source control                                                  | GCS                               |
| pipeline/resources/training_meta.json                   | Kept in repository                           | Training metadata required for reproducibility; not generated output                                  | Repository                        |
| Online_Extraction/extracted_news.json                   | Remove with deprecated folder                | Generated extraction output; not source code                                                          | GCS                               |
| Online_Extraction/partial_all_RSS.json.gz               | Remove with deprecated folder                | Raw collected RSS data; generated artifact                                                            | GCS                               |
| Online_Extraction/tweets.json                           | Remove with deprecated folder                | Generated extraction output; not source code                                                          | GCS                               |
| Model_training/Articles_with_Stories.csv.gz             | Remove with deprecated folder                | Legacy generated artifact; superseded by pipeline version                                             | GCS                               |
| Model_training/BERTopic_changes.csv                     | Remove with deprecated folder                | Legacy generated artifact; superseded by pipeline version                                             | GCS                               |
| Model_training/Canonical_Stories_with_Summaries.csv     | Remove with deprecated folder                | Legacy generated artifact; superseded by pipeline version                                             | GCS                               |
| Model_training/Story_Clusters.csv.gz                    | Remove with deprecated folder                | Legacy generated artifact; superseded by pipeline version                                             | GCS                               |
| Model_training/dashboard_articles.csv.gz                | Remove with deprecated folder                | Legacy generated dashboard artifact                                                                   | GCS / Reporting Layer             |
| Model_training/dashboard_dropdown.csv.gz                | Remove with deprecated folder                | Legacy generated dashboard artifact                                                                   | GCS / Reporting Layer             |
| Model_training/dashboard_stories.csv.gz                 | Remove with deprecated folder                | Legacy generated dashboard artifact                                                                   | GCS / Reporting Layer             |
| Model_training/final_risk_scores1.csv                   | Remove with deprecated folder                | Legacy generated risk intelligence output                                                             | Database / Reporting Layer        |
| Model_training/risk_mlb.pkl                             | Remove with deprecated folder                | Legacy serialized ML artifact                                                                         | Controlled Model Storage          |
| Model_training/risk_predictions.csv                     | Remove with deprecated folder                | Legacy generated prediction output                                                                    | GCS / Database                    |
| Model_training/risks.json                               | Remove with deprecated folder                | Legacy configuration artifact; superseded by active pipeline version                                  | Controlled Config Storage         |
| Model_training/story_centroids.pkl                      | Remove with deprecated folder                | Legacy clustering artifact                                                                            | Controlled Artifact Storage       |
| Model_training/subtopic_centroids.pkl                   | Remove with deprecated folder                | Legacy clustering artifact                                                                            | Controlled Artifact Storage       |
| Model_training/subtopics.csv                            | Remove with deprecated folder                | Legacy generated topic output                                                                         | GCS                               |
| Model_training/test_articles.csv                        | Remove with deprecated folder                | Legacy test dataset                                                                                   | N/A                               |
| Model_training/topic_trend.csv                          | Remove with deprecated folder                | Legacy generated trend output                                                                         | GCS / Reporting Layer             |
| Model_training/topics_BERT.json                         | Remove with deprecated folder                | Legacy topic metadata artifact                                                                        | Controlled Artifact Storage       |
| Model_training/train_articles.csv                       | Remove with deprecated folder                | Legacy training dataset                                                                               | GCS                               |
| Model_training/training_meta.json                       | Remove with deprecated folder                | Legacy metadata artifact; superseded by active version                                                | N/A                               |
| .devcontainer/devcontainer.json                         | Review for removal                           | Legacy development environment configuration; no production dependency identified                     | Remove if unused                  |

## Remaining recommendations