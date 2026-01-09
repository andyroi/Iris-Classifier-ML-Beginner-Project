import pandas as pd # help work with data tables/frames
import matplotlib.pyplot as plt #creates graphs and charts
import seaborn as sns # seaborn for statistical visuals like graphs
from sklearn.datasets import load_iris # data set for learning -> iris flower data

iris = load_iris() #to initialize and load the iris dataset to "iris" variable
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) #create a DataFrame from the iris data

# apply column for numeric specie labels of 0,1,2
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({0: 'setosa', 1:"versicolor", 2:"virginica"}) #mapping names toward the target int value

#df.shape is a tuple of (rows, columns)
print("\n1. Dataset Shape:")
print(f"  Rows:{iris_df.shape[0]}, Columns: {iris_df.shape[1]}") #0 means rows and 1 means columns

print("\n2. First 5 rows:")
print(iris_df.head()) #which spits first 5 rows by defaults

print(" \n3. Data Set info") #as name implies it gives info of the specific columns, the non-null count  and "dtype" which is float64, int64, and object
print(iris_df.info())

print(" \n4. Statistical Summary") # using .describe() it would show calculating mean, std, min, max, etc.
print(iris_df.describe())

print("\n5. Class Distribution:")
print(iris_df['species_name'].value_counts()) # .value_countS() should give me the count occurence of the specific label

#going to visualizng via matplot now (graphs and charts)
print("\n6. Creating visualizations..")
#fig = whole figure, axes = array of indiviudal plots
#2, 2 means 2 rows and 2 columns of the of plots 2x2
# fig size = (12,10 width 12 inches & height 10 inches
fig, axes = plt.subplots(2, 2, figsize=(12,10))
fig.suptitle('Iris Dataset Exploration', fontsize=16)

#apply scatterplot toward first plot via seaborn
sns.scatterplot(
    data=iris_df,
    x='sepal length (cm)', #column for xaxis
    y='sepal width (cm)', #column for yaxis
    hue='species_name', #color based on column species name
    ax=axes[0,0], #beginning position
    palette='Set1'#color scheme to use - colors for different titles
)
axes[0,0].set_title('Sepal Length vs Width') #apply title to the 2x2 of the subplots

#second scatter plot but same pattern x and y are based on the columns with the data
sns.scatterplot(
    data=iris_df, 
    x='petal length (cm)', 
    y='petal width (cm)', 
    hue='species_name', 
    ax=axes[0, 1],  # Position [row 0, column 1]
    palette='Set1'
)
axes[0, 1].set_title('Petal Length vs Width')

#creating box plot for petal
iris_df.boxplot(
    column='petal length (cm)',
    by = 'species_name',
    ax=axes[1,0]
)
axes[1,0].set_title('Petal Length Distribution by Species')
axes[1,0].set_xlabel('Species')

#Creating a correlation heatmap
correlation = iris_df[iris.feature_names].corr()
# iris_df[iris.feature_names] selects jsut the measurement columns
# .corr() calculate thew correlation between all columns
# Correlation shows how related two variables are (-1 to 1)

sns.heatmap(
    correlation, #matrix we created to display
    annot=True, #means shows numbers in each cell
    cmap='coolwarm', #just color map: cool (blue) to warm (red)
    ax=axes[1,1] # last subplot
    
)

plt.tight_layout() #adjust the spacing so it isn't overlapped

#saving the figure as a png image file
plt.savefig('iris_exploration.png')
print("   Saved visualization as 'iris_exploration.png'")

plt.show() #display plots on screen