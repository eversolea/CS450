#PART 1
# 01
import random
import numpy as np
print("part 1: \n")

class Movie:
    def __init__(self, title = "", year = 0, runtime = 0):
        self.title = title
        self.year = year
        self.runtime = runtime
        #06
        if(runtime < 0):
            self.runtime = 0


    def __repr__(self):
        return self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " minutes"
        
    def calcRuntimeHours(self):
        runtimeHours = int(self.runtime / 60)
        runtimeMins = int(self.runtime % 60)
        return (runtimeHours,runtimeMins)
        
# 02
movie1 = Movie()
movie1.title = "Inception"
movie1.year = 2003
movie1.runtime = 190

# 03
print(movie1)

# 04
print(movie1.calcRuntimeHours())

# 05
movie2 = Movie("Jurassic World", 2015, 124)
print(movie2) 

#06
movie3= Movie("Jurassic World", 2015, -34)
print(movie3)


print("\npart 2: \n")
#PART 2
#01
def create_movie_list():
    movieA = Movie("Batman Begins", 2005, 140)
    movieB = Movie("The Dark Knight", 2008, 152)
    movieC = Movie("The Dark Knight Rises", 2012, 164)
    movieD = Movie("Transformers", 2007, 144)
    movieE = Movie("The Song of God", 2014, 138)
    movieList = [movieA, movieB, movieC, movieD, movieE]
    return movieList

  
        


#03
print("\n Movies longer than 150 minutes: \n")
newMovieList = [movie for movie in create_movie_list() if (movie.runtime > 150)]
for i, val in enumerate(newMovieList):
    print(val)

#04
stars_map = dict()
for movie in create_movie_list():
    title = movie.title
    rating = random.uniform(0, 5)
    stars_map[title] = rating
    
    
print("\n What the Viewers say!\n")
#05
for title in stars_map:
    print("{} with a rating of {:.2f}".format(title, stars_map[title]))
    
#PART 3
print("Part 3: ")
 
def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

    
#1.2
def main():
    movieList = create_movie_list()
    for i, val in enumerate(movieList):
        print(val)

    #2.5
    array = get_movie_data()
    #3.2
    print(str(array.shape[0]) + " Rows")
    print(str(array.shape[1]) + " Columns\n")
    
    #3.3
    print("Original array: \n")
    print(array)
    print("\n")
    print(array[0:2,0:3])
    #3.4
    print("\n")
    print(array[0:,1:])
    
    #3.5
    print("\n")
    flatArray = array[0:,1:2].flatten()
    print(flatArray)
    
    
main()