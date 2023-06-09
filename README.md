# Students Clustering
This project is an implementation of K-Means clustering algorithm to segregate students into two groups based on their performance on 2 exams.

## Dataset
The dataset used in this project consists of 100 students and their scores on two exams. It is a CSV file with the following columns:
* Exam 1 
* Exam 2 

## Dependencies
The following libraries are required to run this project:
* NumPy
* Pandas
* Matplotlib


## Usage
To run this project, you can simply clone the repository and run the moving_means.py file.

```
git clone https://github.com/Maximo-Rulli/Students_Clustering.git
```

```
cd kmeans-clustering
```

### Install the required packages:
```
pip install -r requirements.txt
```

### Run the program to visualize the algorithm working:

```
python moving_means.py
```

## Results
The K-Means algorithm was able to segregate the students into two groups based on their exam scores. The following scatter plot shows the two groups of students:

![Scatterplot of students' exams scores](students.png)

The red dots represent students who performed better overall, while the green triangles are students who did not perfom as well as the others overall. The crosses are the clusters' centroids.
