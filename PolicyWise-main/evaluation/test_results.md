# PolicyWise – Evaluation Results

## 1. Testing Summary

The following test cases will be used to evaluate the PolicyWise HR Policy Assistant.

## 2. Test Cases

| Test ID | Test Scenario | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC01 | Tier 1 hotel limit | INR 6000 per night | Pending | Not Tested |
| TC02 | Remote working days | 8 days per calendar month with approval | Pending | Not Tested |
| TC03 | Expense receipt deadline | Within 30 calendar days | Pending | Not Tested |
| TC04 | Business-class approval | Written CFO approval before booking | Pending | Not Tested |
| TC05 | Superseded hotel limit | Identify INR 4500 as an old, superseded limit | Pending | Not Tested |
| TC06 | Personal entertainment expenses | Not reimbursable | Pending | Not Tested |
| TC07 | Personal devices for production access | Not permitted | Pending | Not Tested |
| TC08 | Unsupported question | Exact fallback message | Pending | Not Tested |

## 3. Evaluation Criteria

1. **Answer Accuracy:** Does the answer match the policy document?
2. **Retrieval Relevance:** Were the correct policy sections retrieved?
3. **Source Citation:** Are the PDF filename and page number correct?
4. **Hallucination Prevention:** Does the system avoid unsupported answers?
5. **Version Handling:** Does it prioritize the latest applicable policy?
6. **Fallback Handling:** Does it return the exact fallback for unsupported questions?

## 4. Final Evaluation

**Overall Result:** Pending

Testing will be completed after integration of the document ingestion, retrieval, answer generation, and Streamlit frontend modules.

