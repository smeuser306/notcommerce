package org.invisible.notcommerce.services.implementations;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.invisible.notcommerce.services.RecommendationEngine;
import org.springframework.stereotype.Service;

@Service
public class RecommendationEngineImpl implements RecommendationEngine {
    @Override
    public void trainModel() {
        // 1. Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("Recommendation Engine")
                .master("local[*]") // Run locally with all available cores
                .getOrCreate();

        // 2. Define Schema for Input Data
        StructType schema = new StructType(new StructField[]{
                new StructField("Customer_ID", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("Product_Name", DataTypes.StringType, false, Metadata.empty()),
                new StructField("Rating", DataTypes.IntegerType, false, Metadata.empty())
        });

        // 3. Load Dataset
        Dataset<Row> purchases = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv("src/main/resources/ml-latest-small/purchases.csv"); // Path to the dataset file

        // 4. Display Dataset
        System.out.println("Input Data:");
        purchases.show();

        StringIndexer productIndexer = new StringIndexer()
                .setInputCol("Product_Name")
                .setOutputCol("Product_ID")
                .setHandleInvalid("skip");

        Dataset<Row> indexedData = productIndexer.fit(purchases).transform(purchases);
        indexedData.show();

        // 5. Train ALS Model
        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("Customer_ID")
                .setItemCol("Product_ID")
                .setRatingCol("Rating");

        ALSModel model = als.fit(indexedData);

        // 6. Generate Recommendations for Users
        Dataset<Row> userRecommendations = model.recommendForAllUsers(3); // 3 recommendations per user
        System.out.println("User Recommendations:");
        userRecommendations.show(false);

        // 7. Generate Recommendations for Products
        Dataset<Row> productRecommendations = model.recommendForAllItems(3); // 3 recommendations per product
        System.out.println("Product Recommendations:");
        productRecommendations.show(false);

        // 8. Stop Spark Session
        spark.stop();
    }
}
