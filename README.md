

# The famous Iris flower data set

The Iris flower data set or Fisher's Iris data set is a multivariate data set used and made famous by the British statistician and biologist Ronald Fisher in his 1936 paper _The use of multiple measurements in taxonomic problems_ as an example of linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

![iris examples](img/iris_img.png "Iris examples")

## Project structure

```
├─ img/
├─ notebooks/
├─ src/

```

`img`: images used to document this project <br />
`notebooks`: notebooks used during analysis and model training <br />
`src`: helper functions used during analysis

## Business understanding

The objective of the project is to perform exploratory data analysis (EDA) and train a model capable of classifying a flower into one of three species. This project follows the CRISP-DM framework.

## Data understanding

The dataset contains a set of 150 records under five attributes: sepal length, sepal width, petal length, petal width, and species. In this section, we explored the distribution of data across different species. The number of samples for each of the three species was the same, as we can see in the chart below, meaning we don't have any kind of imbalance between classes.

![species_count](img/species_countplot.png)

The next step was to understand the characteristics and statistics of each feature, including its distribution, mean values, standard deviation, and to check for outlier values. A summary of this can be seen through the boxplot below. As we can see, there are a few outliers in the distribution, but these are real values, which means we cannot discard them as they could also occur in future samples. Also through this chart, it is possible to identify that Iris Setosa has a very different distribution of values for petal length and petal width, with no overlap between its values and those of the other two classes.

![features_boxplot](img/features_boxplot.png)

After that, we looked at the relation between features. In this step, we first created three new features calculated from the existing ones.

<table>
    <tr>
        <th>feature</th>
        <th>observation</th>
    </tr>
    <tr>
        <td>Petal_Prop</td><td>rate between petal length and petal width</td>
    </tr>
     <tr>
        <td>Sepal_Prop</td><td>rate between sepal length and sepal width</td>
    </tr>
         <tr>
        <td>SPL/SL</td><td>rate between petal length and sepal length</td>
    </tr>
</table>

With these features, we explored the relation between each one, for each species. As we can observe, petal length and petal width seem to have a strong relation. Also from the plots, we can clearly see that Iris Setosa can be easily separated from the other two, as it was previously suggested when we analyzed the distribution of our data. 

![features_relation](img/features_scatterplot.png)

We can see how strong the correlation between features is through the correlation matrix. A correlation matrix is a table that displays the correlation coefficients between pairs of variables. This helps visualize the strength and direction (positive or negative) of relationships within a dataset.

![correaltion_matrix](img/correlation_matrix_plot.png)

## Data preparation

To make a streamline process of all the transformations and features engineering we've created a pipeline where the new features as computed.