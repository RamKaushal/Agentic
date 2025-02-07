# Agentic

Generate a realistic daily call volume dataset for a banking client (e.g., Citi) from January 1, 2024, to December 31, 2025. The dataset should reflect:

📈 **Trends & Seasonality**:
- A **gradual yearly increase** in call volume (~10% over two years).
- **Weekly patterns**: Higher volumes on **weekends (Saturday & Sunday)**.
  
📅 **Holidays & Special Events**:
- **Increased** call volume on shopping-related holidays: **Black Friday, Cyber Monday, Christmas Eve, and New Year's Eve**.
- **Decreased** call volume on major holidays: **New Year's Day, Independence Day, Thanksgiving, and Christmas**.
- Adjust call volumes on **days before and after major holidays** to reflect customer behavior.

💳 **Billing Date Impact**:
- **Higher call volume** on **billing dates** (7th, 25th, 26th, and 28th of each month).

⚠ **Data Anomalies**:
- **Random missing days**, null values, and **outliers** to simulate real-world inconsistencies.
- **System outages** or **unexpected fluctuations** as reasons for anomalies.

🎯 **Feature Engineering for AI Models (e.g., XGBoost)**:
- **Date-Based Features**:
  - **Day of the Week** (0 = Monday, 6 = Sunday)
  - **Day of the Month**
  - **Day of the Year**
  - **Week of the Year**
  - **Month of the Year**
  - **Quarter of the Year**
  - **Is Weekend (Binary: 1 = Yes, 0 = No)**
- **Holiday Indicator**: Specify the holiday name if applicable.
- **Call Volume Impact**: Explanation of volume fluctuations (e.g., 'Increased due to Black Friday', 'Decreased due to New Year's Day').

📊 **Output Format**:
- **Columns**: `Date`, `Call Volume`, `U.S. Holiday Indicator`, `Call Volume Impact`, `Day of the Week`, `Day of the Month`, `Day of the Year`, `Week of the Year`, `Month`, `Quarter`, `Is Weekend`
- `Call Volume` should be a **numeric value** reflecting the above patterns.

📌 **Ensure that the generated dataset realistically follows these trends while incorporating randomness to simulate real-world data.**
