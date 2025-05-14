import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
from sentiment_analysis import preprocess_data, analyze_sentiment



# Streamlit Page Config
st.set_page_config(page_title="SentimentX : E-Commerce Sentiment Analysis", layout="wide")

st.title("🛒 SentimentX : AI-Powered Sentiment Analysis for E-Commerce Sellers")

# File Upload# format{"Productname","Sentiments","Ratings","Timestamps"}
uploaded_file = st.file_uploader("Upload a CSV file with format {Productname,Sentiments,Ratings,Timestamps}", type=["csv"])


if uploaded_file:
    try:
        # Load and analyze data
        df = preprocess_data(uploaded_file)
        df = analyze_sentiment(df)

        # Ensure timestamps are in datetime format (if available)
        if "timestamp" in df.columns:
           df["timestamp"] = pd.to_datetime(df["timestamp"])


        # Dropdown for selecting a question
        st.subheader("🔍 Select an Analysis Question:")
        question = st.radio(
            "Choose a question to analyze:",
            [
                "1. Which is the top product preferred by customers?",
                "2. Which product has the most 5-star ratings?",
                "3. Which product is the worst?",
                "4. Which are the top five products?",
                "5. Analysis for each product based on ratings ?",
                "6. Which product has received the most reviews?",
                "7. Which product has the most negative reviews?",
                "8. What is the Overall product customer sentiment trend?",
                "9. Which product has the most mixed (neutral) reviews?",
                "10. Sentiment trend over time (if timestamps are available)?",
                "11. Which product has the most customer engagement?",
                "12. Product Trendline improved over time?", 
                "13. Sentimental Analysis For each Product : " # Requires timestamp
            ],
            index=None  # No default selection
        )

        # Ensure a question is selected before proceeding
        if question:

            if question == "1. Which is the top product preferred by customers?":
                st.subheader("🏆 Top Product Preferred by Customers")
                top_product = df.groupby("productname")["sentiment_score"].mean().idxmax()
                st.success(f"**{top_product}** is the most preferred product based on customer sentiment.")
                

            elif question == "2. Which product has the most 5-star ratings?":
                st.subheader("⭐ Product with Most 5-Star Ratings")
                top_rated_product = df[df["ratings"] == 5]["productname"].value_counts().idxmax()
                st.success(f"**{top_rated_product}** has received the most 5-star ratings.")

            elif question == "3. Which product is the worst?":
                st.subheader("❌ Worst Product")
                worst_product = df.groupby("productname")["sentiment_score"].mean().idxmin()
                st.error(f"**{worst_product}** is rated as the worst product by customers.")

            elif question == "4. Which are the top five products?":
                st.subheader("📊 Top 5 Products by Sentiment Score")
                top_five = df.groupby("productname")["sentiment_score"].mean().sort_values(ascending=False).head(5)

                fig, ax = plt.subplots()
                sns.barplot(x=top_five.values, y=top_five.index, ax=ax,color="#F78000")
                ax.set_title("Top 5 Products by Sentiment Score")
                st.pyplot(fig)
                st.success("These are the top-performing products based on sentiment scores.")
                

               


            

            elif question == "5. Analysis for each product based on ratings ?":
                st.subheader("📈 Sentiment Breakdown for Each Product")
                sentiment_counts = df.groupby("productname")["ratings"].value_counts().unstack()

                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot(kind="bar", stacked=True, ax=ax, colormap="inferno")
                ax.set_title("Ratings Breakdown for Each Product")
                ax.set_ylabel("Count")
                st.pyplot(fig)
                st.success("This shows the proportion of positive, negative, and neutral reviews for each product.")

            elif question == "6. Which product has received the most reviews?":
                st.subheader("📢 Most Reviewed Product")
                most_reviewed_product = df["productname"].value_counts().idxmax()

                fig, ax = plt.subplots()
                df["productname"].value_counts().head(10).plot(kind="bar", ax=ax,color="#F78000")
                ax.set_title("Top 10 Most Reviewed Products")
                st.pyplot(fig)
                st.success(f"**{most_reviewed_product}** has received the highest number of reviews.")

            elif question == "7. Which product has the most negative reviews?":
                st.subheader("❌ Most Negative Product")
                most_negative_product = df[df["sentiment_category"] == "Negative"]["productname"].value_counts().idxmax()

                fig, ax = plt.subplots()
                df[df["sentiment_category"] == "Negative"]["productname"].value_counts().head(10).plot(kind="bar", ax=ax, color="red")
                ax.set_title("Top 10 Most Negative Products")
                st.pyplot(fig)
                st.error(f"**{most_negative_product}** has received the most negative reviews.")

            elif question == "8. What is the Overall product customer sentiment trend?":
                st.subheader("📊 Customer Sentiment Trend Per Product")
                sentiment_counts = df["sentiment_category"].value_counts()

                fig, ax = plt.subplots()
                sentiment_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=90, colors=["green", "red", "gray"])
                ax.set_ylabel("")
                ax.set_title("Overall Customer Sentiment Distribution")
                st.pyplot(fig)
                st.success("This shows the overall distribution of positive, negative, and neutral sentiments.")

            elif question == "9. Which product has the most mixed (neutral) reviews?":
                st.subheader("🟠 Product with Most Neutral Reviews")
                
                # Filter neutral reviews
                neutral_reviews = df[df["sentiment_category"] == "Neutral"]
                
                if not neutral_reviews.empty:
                    most_neutral_product = neutral_reviews["productname"].value_counts().idxmax()
                    st.success(f"**{most_neutral_product}** has received the highest number of neutral reviews.")
                else:
                    st.warning("No neutral reviews found in the dataset.")
            elif question == "10. Sentiment trend over time (if timestamps are available)?":
                if "timestamp" in df.columns:
                    st.subheader("📈 Sentiment Trend Over Time")
                    
                    # Adding a dropdown to select granularity (Year, Month, Day)
                    time_granularity = st.selectbox("Select time granularity:", ["Year", "Month", "Day"])

                    # Convert timestamp to datetime (if not already in datetime format)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    if time_granularity == "Year":
                        # Group by year and calculate mean sentiment
                        df_time = df.groupby(df["timestamp"].dt.year)["sentiment_score"].mean()
                    elif time_granularity == "Month":
                        # Group by year and month (for better granularity)
                        df_time = df.groupby(df["timestamp"].dt.to_period("M"))["sentiment_score"].mean()
                    else:  # For Day
                        # Group by day and calculate mean sentiment
                        df_time = df.groupby(df["timestamp"].dt.date)["sentiment_score"].mean()

                    # Plotting the trendline
                    fig, ax = plt.subplots(figsize=(8, 5))  # You can adjust the size as per your need
                    df_time.plot(kind="line", marker="o", ax=ax)

                    # Set axis labels based on the selected time granularity
                    ax.set_title(f"Customer Sentiment Trend Over Time ({time_granularity})")
                    ax.set_xlabel(time_granularity)
                    ax.set_ylabel("Average Sentiment Score")

                    st.pyplot(fig)
                    st.success("This shows how customer sentiment has evolved over time.")
                else:
                    st.error("Timestamp data not found in the uploaded file.")

            elif question == "11. Which product has the most customer engagement?":
                st.subheader("📢 Most Engaging Product")
                most_engaging_product = df["productname"].value_counts().idxmax()
                st.success(f"**{most_engaging_product}** has received the most total reviews, indicating high customer engagement.")

            elif question == "12. Product Trendline improved over time?":
               
                st.subheader("📈 Sentiment Trend for a Specific Product in the Current Year")

                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Automatically detect current year
              
                # Get unique product names
                products = df["productname"].unique()
                available_years = df["timestamp"].dt.year.unique()
                # Product selection dropdown
                selected_product = st.selectbox("Select a product to analyze:", products)
                current_year = st.selectbox("Select a year to analyze:", sorted(available_years, reverse=True))

                # Filter data for selected product and current year
                filtered_df = df[(df["productname"] == selected_product) & 
                                (df["timestamp"].dt.year == current_year)]

                # Check if there's data to plot
                if not filtered_df.empty:
                    # Group by month and calculate average sentiment
                    monthly_trend = filtered_df.groupby(filtered_df["timestamp"].dt.month)["sentiment_score"].mean()

                    # Plot the trendline
                    fig, ax = plt.subplots(figsize=(8,5))
                    monthly_trend.plot(kind='line', marker='o', ax=ax)
                    ax.set_title(f"Sentiment Trend Over Months - {selected_product} ({current_year})")
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Average Sentiment Score")
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.warning(f"No data found for {selected_product} in {current_year}.")
            elif question == "13. Sentimental Analysis For each Product : ":
                st.subheader("📊 Sentimental Analysis For Each Product")
                product_sentiments = df.groupby("productname")["sentiment_category"].value_counts().unstack().fillna(0)
                for product, sentiments in product_sentiments.iterrows():
                    fig, ax = plt.subplots()
                    sentiments.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
                    ax.set_ylabel("")
                    ax.set_title(f"Sentiment Analysis for {product}")
                    st.pyplot(fig)
            


    except Exception as e:
        st.error(f"Error: {e}")
