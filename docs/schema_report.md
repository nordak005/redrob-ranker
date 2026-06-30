# Redrob Candidate Schema Exploration & Dataset Profile

This document profiles the 100K candidate dataset and acts as the blueprint for Phase 2 ranking.

## 1. Executive Summary
- **Total Candidates**: 100,000
- **Nested Schema Max Depth**: 7
- **Anomalous (Potential Honeypot) Records**: 45 candidates detected with impossible profiles

## 2. Field Analysis & Types
| Field Path | Detected Types | Count | Null Rate (%) | Depth | Examples |
| --- | --- | --- | --- | --- | --- |
| `` | dict | 100,000 | 0.00% | 0 |  |
| `candidate_id` | str | 100,000 | 0.00% | 0 | 'CAND_0000001', 'CAND_0000002', 'CAND_0000003' |
| `career_history` | list | 100,000 | 0.00% | 0 |  |
| `career_history[*]` | dict | 300,171 | -200.17% | 1 |  |
| `career_history[*].company` | str | 300,171 | -200.17% | 2 | 'Mindtree', 'Dunder Mifflin', 'Wipro' |
| `career_history[*].company_size` | str | 300,171 | -200.17% | 2 | '10001+', '201-500', '501-1000' |
| `career_history[*].description` | str | 300,171 | -200.17% | 2 | 'Implemented streaming data pipelines on Kafka and Spark Str |
| `career_history[*].duration_months` | int | 300,171 | -200.17% | 2 | 27, 55, 43 |
| `career_history[*].end_date` | str | 200,171 | -100.17% | 2 | '2024-01-08', '2022-11-07', '2021-08-14' |
| `career_history[*].industry` | str | 300,171 | -200.17% | 2 | 'IT Services', 'Paper Products', 'Manufacturing' |
| `career_history[*].is_current` | bool | 300,171 | -200.17% | 2 | True, False |
| `career_history[*].start_date` | str | 300,171 | -200.17% | 2 | '2024-03-08', '2019-07-03', '2022-11-14' |
| `career_history[*].title` | str | 300,171 | -200.17% | 2 | 'Backend Engineer', 'Analytics Engineer', 'Operations Manage |
| `certifications` | list | 100,000 | 0.00% | 0 |  |
| `certifications[*]` | dict | 37,484 | 62.52% | 1 |  |
| `certifications[*].issuer` | str | 37,484 | 62.52% | 2 | 'AWS', 'Scrum Alliance', 'ASQ' |
| `certifications[*].name` | str | 37,484 | 62.52% | 2 | 'AWS Certified Cloud Practitioner', 'Scrum Master Certified' |
| `certifications[*].year` | int | 37,484 | 62.52% | 2 | 2025, 2018, 2019 |
| `education` | list | 100,000 | 0.00% | 0 |  |
| `education[*]` | dict | 139,778 | -39.78% | 1 |  |
| `education[*].degree` | str | 139,778 | -39.78% | 2 | 'B.E.', 'B.Sc', 'M.E.' |
| `education[*].end_year` | int | 139,778 | -39.78% | 2 | 2020, 2011, 2010 |
| `education[*].field_of_study` | str | 139,778 | -39.78% | 2 | 'Computer Science', 'Mathematics', 'Chemical Engineering' |
| `education[*].grade` | str | 139,778 | -39.78% | 2 | '8.24 CGPA', '77%', '6.82 CGPA' |
| `education[*].institution` | str | 139,778 | -39.78% | 2 | 'Lovely Professional University', 'Local Engineering College |
| `education[*].start_year` | int | 139,778 | -39.78% | 2 | 2017, 2007, 2005 |
| `education[*].tier` | str | 139,778 | -39.78% | 2 | 'tier_3', 'tier_4', 'tier_2' |
| `languages` | list | 100,000 | 0.00% | 0 |  |
| `languages[*]` | dict | 200,000 | -100.00% | 1 |  |
| `languages[*].language` | str | 200,000 | -100.00% | 2 | 'English', 'Hindi' |
| `languages[*].proficiency` | str | 200,000 | -100.00% | 2 | 'professional', 'conversational', 'native' |
| `profile` | dict | 100,000 | 0.00% | 0 |  |
| `profile.anonymized_name` | str | 100,000 | 0.00% | 1 | 'Ira Vora', 'Saanvi Sethi', 'Yash Agarwal' |
| `profile.country` | str | 100,000 | 0.00% | 1 | 'Canada', 'India', 'USA' |
| `profile.current_company` | str | 100,000 | 0.00% | 1 | 'Mindtree', 'Wipro', 'TCS' |
| `profile.current_company_size` | str | 100,000 | 0.00% | 1 | '10001+', '201-500', '1001-5000' |
| `profile.current_industry` | str | 100,000 | 0.00% | 1 | 'IT Services', 'Paper Products', 'Manufacturing' |
| `profile.current_title` | str | 100,000 | 0.00% | 1 | 'Backend Engineer', 'Operations Manager', 'Customer Support' |
| `profile.headline` | str | 100,000 | 0.00% | 1 | 'Backend Engineer | SQL, Spark, Cloud', 'Operations Manager  |
| `profile.location` | str | 100,000 | 0.00% | 1 | 'Toronto', 'Chennai, Tamil Nadu', 'Austin' |
| `profile.summary` | str | 100,000 | 0.00% | 1 | "Software / data professional with 6.9 years of experience b |
| `profile.years_of_experience` | float | 100,000 | 0.00% | 1 | 6.9, 12.5, 1.1 |
| `redrob_signals` | dict | 100,000 | 0.00% | 0 |  |
| `redrob_signals.applications_submitted_30d` | int | 100,000 | 0.00% | 1 | 2, 1, 9 |
| `redrob_signals.avg_response_time_hours` | float | 100,000 | 0.00% | 1 | 177.8, 171.6, 119.4 |
| `redrob_signals.connection_count` | int | 100,000 | 0.00% | 1 | 356, 179, 19 |
| `redrob_signals.endorsements_received` | int | 100,000 | 0.00% | 1 | 35, 3, 46 |
| `redrob_signals.expected_salary_range_inr_lpa` | dict | 100,000 | 0.00% | 1 |  |
| `redrob_signals.expected_salary_range_inr_lpa.max` | float | 100,000 | 0.00% | 2 | 36.1, 9.0, 18.1 |
| `redrob_signals.expected_salary_range_inr_lpa.min` | float | 100,000 | 0.00% | 2 | 18.7, 8.8, 11.2 |
| `redrob_signals.github_activity_score` | float, int | 100,000 | 0.00% | 1 | 9.2, -1, 33.7 |
| `redrob_signals.interview_completion_rate` | float | 100,000 | 0.00% | 1 | 0.71, 0.62, 0.86 |
| `redrob_signals.last_active_date` | str | 100,000 | 0.00% | 1 | '2026-05-20', '2025-11-12', '2026-03-21' |
| `redrob_signals.linkedin_connected` | bool | 100,000 | 0.00% | 1 | False, True |
| `redrob_signals.notice_period_days` | int | 100,000 | 0.00% | 1 | 60, 150, 120 |
| `redrob_signals.offer_acceptance_rate` | float, int | 100,000 | 0.00% | 1 | 0.58, -1, 0.48 |
| `redrob_signals.open_to_work_flag` | bool | 100,000 | 0.00% | 1 | True, False |
| `redrob_signals.preferred_work_mode` | str | 100,000 | 0.00% | 1 | 'onsite', 'flexible', 'hybrid' |
| `redrob_signals.profile_completeness_score` | float | 100,000 | 0.00% | 1 | 86.9, 78.7, 31.9 |
| `redrob_signals.profile_views_received_30d` | int | 100,000 | 0.00% | 1 | 23, 7, 1 |
| `redrob_signals.recruiter_response_rate` | float | 100,000 | 0.00% | 1 | 0.34, 0.29, 0.46 |
| `redrob_signals.saved_by_recruiters_30d` | int | 100,000 | 0.00% | 1 | 4, 10, 8 |
| `redrob_signals.search_appearance_30d` | int | 100,000 | 0.00% | 1 | 249, 107, 28 |
| `redrob_signals.signup_date` | str | 100,000 | 0.00% | 1 | '2025-10-16', '2025-07-28', '2024-08-02' |
| `redrob_signals.skill_assessment_scores` | dict | 100,000 | 0.00% | 1 |  |
| `redrob_signals.skill_assessment_scores.ASR` | float | 1,124 | 98.88% | 2 | 53.6, 58.6, 67.7 |
| `redrob_signals.skill_assessment_scores.BM25` | float | 309 | 99.69% | 2 | 50.0, 59.0, 44.8 |
| `redrob_signals.skill_assessment_scores.BentoML` | float | 1,157 | 98.84% | 2 | 74.5, 27.8, 80.3 |
| `redrob_signals.skill_assessment_scores.CNN` | float | 1,174 | 98.83% | 2 | 54.8, 46.6, 68.4 |
| `redrob_signals.skill_assessment_scores.Computer Vision` | float | 1,111 | 98.89% | 2 | 34.8, 64.9, 68.9 |
| `redrob_signals.skill_assessment_scores.Data Science` | float | 1,147 | 98.85% | 2 | 35.1, 23.9, 47.8 |
| `redrob_signals.skill_assessment_scores.Deep Learning` | float | 284 | 99.72% | 2 | 74.3, 45.3, 79.5 |
| `redrob_signals.skill_assessment_scores.Diffusion Models` | float | 1,133 | 98.87% | 2 | 70.6, 36.9, 37.4 |
| `redrob_signals.skill_assessment_scores.Elasticsearch` | float | 285 | 99.72% | 2 | 76.0, 35.9, 37.0 |
| `redrob_signals.skill_assessment_scores.Embeddings` | float | 327 | 99.67% | 2 | 41.5, 70.3, 83.8 |
| `redrob_signals.skill_assessment_scores.FAISS` | float | 303 | 99.70% | 2 | 77.6, 68.4, 54.5 |
| `redrob_signals.skill_assessment_scores.Feature Engineering` | float | 1,174 | 98.83% | 2 | 60.8, 39.7, 72.2 |
| `redrob_signals.skill_assessment_scores.Fine-tuning LLMs` | float | 282 | 99.72% | 2 | 41.6, 42.1, 31.0 |
| `redrob_signals.skill_assessment_scores.Forecasting` | float | 1,167 | 98.83% | 2 | 65.1, 64.3, 30.0 |
| `redrob_signals.skill_assessment_scores.GANs` | float | 1,122 | 98.88% | 2 | 53.3, 73.0, 64.2 |
| `redrob_signals.skill_assessment_scores.Haystack` | float | 314 | 99.69% | 2 | 64.5, 73.9, 81.7 |
| `redrob_signals.skill_assessment_scores.Hugging Face Transformers` | float | 319 | 99.68% | 2 | 64.3, 77.5, 65.7 |
| `redrob_signals.skill_assessment_scores.Image Classification` | float | 1,120 | 98.88% | 2 | 64.8, 57.1, 30.2 |
| `redrob_signals.skill_assessment_scores.Information Retrieval` | float | 343 | 99.66% | 2 | 84.7, 75.5, 56.9 |
| `redrob_signals.skill_assessment_scores.Kubeflow` | float | 1,105 | 98.89% | 2 | 57.6, 44.0, 30.7 |
| `redrob_signals.skill_assessment_scores.LLMs` | float | 323 | 99.68% | 2 | 39.3, 56.2, 48.6 |
| `redrob_signals.skill_assessment_scores.LangChain` | float | 295 | 99.70% | 2 | 40.0, 33.2, 42.1 |
| `redrob_signals.skill_assessment_scores.Learning to Rank` | float | 327 | 99.67% | 2 | 77.7, 67.4, 75.3 |
| `redrob_signals.skill_assessment_scores.LlamaIndex` | float | 291 | 99.71% | 2 | 76.6, 55.6, 42.0 |
| `redrob_signals.skill_assessment_scores.LoRA` | float | 320 | 99.68% | 2 | 47.0, 61.3, 69.6 |
| `redrob_signals.skill_assessment_scores.MLOps` | float | 1,159 | 98.84% | 2 | 73.4, 50.1, 68.6 |
| `redrob_signals.skill_assessment_scores.MLflow` | float | 1,085 | 98.91% | 2 | 75.1, 74.4, 66.2 |
| `redrob_signals.skill_assessment_scores.Machine Learning` | float | 335 | 99.67% | 2 | 74.0, 45.5, 43.9 |
| `redrob_signals.skill_assessment_scores.Milvus` | float | 340 | 99.66% | 2 | 47.1, 75.7, 36.9 |
| `redrob_signals.skill_assessment_scores.NLP` | float | 306 | 99.69% | 2 | 38.8, 65.9, 53.1 |
| `redrob_signals.skill_assessment_scores.Object Detection` | float | 1,145 | 98.86% | 2 | 81.3, 41.0, 37.5 |
| `redrob_signals.skill_assessment_scores.OpenCV` | float | 1,155 | 98.84% | 2 | 65.5, 47.4, 34.8 |
| `redrob_signals.skill_assessment_scores.OpenSearch` | float | 319 | 99.68% | 2 | 71.7, 76.2, 72.7 |
| `redrob_signals.skill_assessment_scores.PEFT` | float | 341 | 99.66% | 2 | 50.5, 69.9, 57.3 |
| `redrob_signals.skill_assessment_scores.Pinecone` | float | 325 | 99.67% | 2 | 53.6, 25.4, 78.1 |
| `redrob_signals.skill_assessment_scores.Prompt Engineering` | float | 312 | 99.69% | 2 | 73.8, 47.5, 38.0 |
| `redrob_signals.skill_assessment_scores.PyTorch` | float | 340 | 99.66% | 2 | 64.5, 30.5, 49.0 |
| `redrob_signals.skill_assessment_scores.Python` | float | 312 | 99.69% | 2 | 68.1, 76.9, 45.4 |
| `redrob_signals.skill_assessment_scores.QLoRA` | float | 329 | 99.67% | 2 | 75.2, 76.2, 25.1 |
| `redrob_signals.skill_assessment_scores.Qdrant` | float | 336 | 99.66% | 2 | 54.4, 47.5, 28.5 |
| `redrob_signals.skill_assessment_scores.RAG` | float | 326 | 99.67% | 2 | 68.4, 84.7, 45.6 |
| `redrob_signals.skill_assessment_scores.Recommendation Systems` | float | 325 | 99.67% | 2 | 29.8, 50.8, 42.2 |
| `redrob_signals.skill_assessment_scores.Reinforcement Learning` | float | 1,124 | 98.88% | 2 | 69.8, 38.1, 57.6 |
| `redrob_signals.skill_assessment_scores.Semantic Search` | float | 301 | 99.70% | 2 | 65.2, 32.0, 70.9 |
| `redrob_signals.skill_assessment_scores.Sentence Transformers` | float | 328 | 99.67% | 2 | 73.1, 65.2, 73.7 |
| `redrob_signals.skill_assessment_scores.Speech Recognition` | float | 1,159 | 98.84% | 2 | 53.7, 62.9, 35.3 |
| `redrob_signals.skill_assessment_scores.Statistical Modeling` | float | 1,088 | 98.91% | 2 | 54.4, 56.8, 27.8 |
| `redrob_signals.skill_assessment_scores.TTS` | float | 1,140 | 98.86% | 2 | 70.2, 27.4, 67.9 |
| `redrob_signals.skill_assessment_scores.TensorFlow` | float | 326 | 99.67% | 2 | 73.5, 36.9, 39.7 |
| `redrob_signals.skill_assessment_scores.Time Series` | float | 1,111 | 98.89% | 2 | 65.0, 35.9, 69.5 |
| `redrob_signals.skill_assessment_scores.Vector Search` | float | 326 | 99.67% | 2 | 73.8, 52.6, 42.6 |
| `redrob_signals.skill_assessment_scores.Weaviate` | float | 321 | 99.68% | 2 | 60.3, 74.1, 31.7 |
| `redrob_signals.skill_assessment_scores.Weights & Biases` | float | 1,173 | 98.83% | 2 | 53.7, 36.0, 37.5 |
| `redrob_signals.skill_assessment_scores.YOLO` | float | 1,195 | 98.80% | 2 | 60.2, 68.9, 72.7 |
| `redrob_signals.skill_assessment_scores.pgvector` | float | 340 | 99.66% | 2 | 43.2, 50.4, 43.3 |
| `redrob_signals.skill_assessment_scores.scikit-learn` | float | 317 | 99.68% | 2 | 68.9, 31.8, 55.6 |
| `redrob_signals.verified_email` | bool | 100,000 | 0.00% | 1 | True, False |
| `redrob_signals.verified_phone` | bool | 100,000 | 0.00% | 1 | True, False |
| `redrob_signals.willing_to_relocate` | bool | 100,000 | 0.00% | 1 | False, True |
| `skills` | list | 100,000 | 0.00% | 0 |  |
| `skills[*]` | dict | 960,302 | -860.30% | 1 |  |
| `skills[*].duration_months` | int | 960,302 | -860.30% | 2 | 13, 26, 40 |
| `skills[*].endorsements` | int | 960,302 | -860.30% | 2 | 3, 37, 7 |
| `skills[*].name` | str | 960,302 | -860.30% | 2 | 'Tailwind', 'NLP', 'Image Classification' |
| `skills[*].proficiency` | str | 960,302 | -860.30% | 2 | 'intermediate', 'advanced', 'beginner' |

## 3. Experience and Career Distributions
### Years of Experience (Profile-level)
- **Minimum**: 1.0 years
- **Maximum**: 16.9 years
- **Mean**: 7.17 years
- **Median**: 6.8 years

### Career History Lengths
- **Mean Job Entries Per Candidate**: 3.00
- **Maximum Job Entries Per Candidate**: 9

## 4. Skills Profiling
- **Mean Skills Listed Per Candidate**: 9.60
- **Maximum Skills Listed Per Candidate**: 23

### Top 20 Most Frequent Skills
| Rank | Skill Name | Occurrences | Frequency (%) |
| --- | --- | --- | --- |
| 1 | HTML | 12,246 | 12.25% |
| 2 | Databricks | 12,244 | 12.24% |
| 3 | Redux | 12,222 | 12.22% |
| 4 | Terraform | 12,187 | 12.19% |
| 5 | Angular | 12,173 | 12.17% |
| 6 | Figma | 12,157 | 12.16% |
| 7 | Salesforce CRM | 12,157 | 12.16% |
| 8 | Vue.js | 12,142 | 12.14% |
| 9 | Sales | 12,138 | 12.14% |
| 10 | Accounting | 12,136 | 12.14% |
| 11 | Agile | 12,135 | 12.13% |
| 12 | Kafka | 12,114 | 12.11% |
| 13 | Excel | 12,109 | 12.11% |
| 14 | BigQuery | 12,108 | 12.11% |
| 15 | CI/CD | 12,108 | 12.11% |
| 16 | Project Management | 12,106 | 12.11% |
| 17 | Airflow | 12,105 | 12.11% |
| 18 | AWS | 12,104 | 12.10% |
| 19 | Flask | 12,104 | 12.10% |
| 20 | Scrum | 12,083 | 12.08% |

## 5. Job Titles Profiling
### Top 20 Job Titles (Current & Historic)
| Rank | Job Title | Occurrences |
| --- | --- | --- |
| 1 | Business Analyst | 24,875 |
| 2 | Mechanical Engineer | 24,783 |
| 3 | Project Manager | 24,770 |
| 4 | Accountant | 24,719 |
| 5 | Graphic Designer | 24,707 |
| 6 | HR Manager | 24,705 |
| 7 | Customer Support | 24,592 |
| 8 | Civil Engineer | 24,584 |
| 9 | Operations Manager | 24,543 |
| 10 | Content Writer | 24,513 |
| 11 | Sales Executive | 24,408 |
| 12 | Marketing Manager | 24,317 |
| 13 | Software Engineer | 11,531 |
| 14 | Full Stack Developer | 9,705 |
| 15 | Java Developer | 9,585 |
| 16 | Cloud Engineer | 9,568 |
| 17 | .NET Developer | 9,495 |
| 18 | Mobile Developer | 9,479 |
| 19 | DevOps Engineer | 9,336 |
| 20 | QA Engineer | 9,251 |

## 6. Location Distribution
### Top 20 Candidate Locations
| Rank | Location | Occurrences | Frequency (%) |
| --- | --- | --- |
| 1 | Bhubaneswar, Odisha | 4,321 | 4.32% |
| 2 | Noida, Uttar Pradesh | 4,283 | 4.28% |
| 3 | Hyderabad, Telangana | 4,283 | 4.28% |
| 4 | Jaipur, Rajasthan | 4,268 | 4.27% |
| 5 | Bangalore, Karnataka | 4,238 | 4.24% |
| 6 | Kolkata, West Bengal | 4,230 | 4.23% |
| 7 | Indore, Madhya Pradesh | 4,198 | 4.20% |
| 8 | Pune, Maharashtra | 4,186 | 4.19% |
| 9 | Chennai, Tamil Nadu | 4,164 | 4.16% |
| 10 | Delhi, Delhi | 4,161 | 4.16% |
| 11 | Trivandrum, Kerala | 4,151 | 4.15% |
| 12 | Ahmedabad, Gujarat | 4,143 | 4.14% |
| 13 | Chandigarh, Chandigarh | 4,128 | 4.13% |
| 14 | Coimbatore, Tamil Nadu | 4,113 | 4.11% |
| 15 | Vizag, Andhra Pradesh | 4,093 | 4.09% |
| 16 | Kochi, Kerala | 4,073 | 4.07% |
| 17 | Mumbai, Maharashtra | 4,043 | 4.04% |
| 18 | Gurgaon, Haryana | 4,037 | 4.04% |
| 19 | Sydney | 2,579 | 2.58% |
| 20 | San Francisco | 2,536 | 2.54% |

## 7. Potential Honeypot / Trap Analysis
A honeypot is defined as a record with impossible combinations of skills/experience:
- **Expert proficiency** listed with **0 months duration** used.
- Profile **years_of_experience** is highly inconsistent with job history (difference > 10 years).

Total anomalies flagged: **45** candidates.

### Sample flagged candidates:
| Candidate ID | Profile Experience | History Experience (Years) | Anomaly Reason |
| --- | --- | --- | --- |
| CAND_0003430 | 13.7 | 0.9 | Profile experience (13.7) differs from history (0.9 years) |
| CAND_0003582 | 8.2 | 8.2 | Expert with 0 duration in: MLflow, Photoshop, Content Writing |
| CAND_0005291 | 12.8 | 0.9 | Profile experience (12.8) differs from history (0.9 years) |
| CAND_0007353 | 9.9 | 20.9 | Profile experience (9.9) differs from history (20.9 years) |
| CAND_0007413 | 13.3 | 1.3 | Profile experience (13.3) differs from history (1.3 years) |
| CAND_0008960 | 10.3 | 22.6 | Profile experience (10.3) differs from history (22.6 years) |
| CAND_0010294 | 8.0 | 18.3 | Profile experience (8.0) differs from history (18.3 years) |
| CAND_0016000 | 2.0 | 2.0 | Expert with 0 duration in: TypeScript, Go, Docker, Hadoop, Photoshop |
| CAND_0024752 | 14.9 | 0.7 | Profile experience (14.9) differs from history (0.7 years) |
| CAND_0025579 | 12.9 | 1.0 | Profile experience (12.9) differs from history (1.0 years) |
