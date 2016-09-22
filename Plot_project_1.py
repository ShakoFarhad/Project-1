import numpy as np
import matplotlib.pyplot as plt

file1 = open('plotGeneralAlgorithm10.txt', 'r') # Here we open the file we want to read
lines1 = file1.readline()    					# Here we read the line in the file and make a list with the one long string containing the elements
lines1 = np.array(lines1.split()).astype(np.float) # Now we split up the string into many small strings where each string contains one number from the data. We then convert all the elements in the list from string to float, so we have a list with datanumbers we can use to plot. We also convert the list into an array
file1.closed # Here we close the file.

file2 = open('plotGeneralAlgorithm100.txt', 'r')
lines2 = file2.readline()
lines2 = np.array(lines2.split()).astype(np.float)
file2.closed

file3 = open('plotGeneralAlgorithm1000.txt', 'r')
lines3 = file3.readline()
lines3 = np.array(lines3.split()).astype(np.float)
file3.closed


x = np.linspace(0,1,1000)							# we create an array containing 1000 elements 
uExact = 1. - (1. - np.exp(-10))*x - np.exp(-10*x)	# here we calculate the exact u-value for every x-value in the array x, and return an array uExact with all the values for u(x).

x1 = np.linspace(0,1,len(lines1)) # x1, x2 and x3, are grid-arras containing an amount of elements corresponding to the amount in the respective dataset.
x2 = np.linspace(0,1,len(lines2))
x3 = np.linspace(0,1,len(lines3))

# The next functions, we plot the results for the three different matrix sizes. We also plot in the same figures, the exact u-values so we can compare the dataresult with the exact values.

plt.figure(1)
plt.plot(x1,lines1,'r-', label='The general algorithm')
plt.plot(x, uExact, 'b-', label='Exact solution for u(x)')
plt.xlabel('Grid points')
plt.ylabel('u(x)')
plt.title('10x10 matrix')
plt.legend()

plt.figure(2)
plt.plot(x2,lines2,'r-', label='The general algorithm')
plt.plot(x, uExact, 'b-', label='Exact solution for u(x)')
plt.xlabel('Grid points')
plt.ylabel('u(x)')
plt.title('100x100 matrix')
plt.legend()

plt.figure(3)
plt.plot(x3,lines3,'r-', label='The general algorithm')
plt.plot(x, uExact, 'b-', label='Exact solution for u(x)')
plt.xlabel('Grid points')
plt.ylabel('u(x)')
plt.title('1000x1000 matrix')
plt.legend()

plt.show()