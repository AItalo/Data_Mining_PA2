import matplotlib.pyplot as plt
import numpy as np

import sys

'''
------------------------------------------------------------------------
HELPER FUNCTIONS
------------------------------------------------------------------------
'''

def process_data(data, column):
    x_i = []
    for row in data:
        x_i.append(row[column])
    xs = create_unique(x_i)
    ys = []
    for x in xs:
        ys.append(get_count(x_i, x))
    return xs, ys


def create_unique(data):
    unique = []
    for x in data:
        if x not in unique:
            unique.append(x)
    unique.sort()
    return unique


def get_count(data, elem):
    count = 0
    for x in data:
        if x == elem:
            count += 1
    return count



def convert_dataset(file_path):
    f = open(file_path, "r")
    dataset = []
    line = f.readline()
    while (line != ''):
        instance_r = line.strip().split(",")
        instance = []
        for elem in instance_r:
            try:
                instance.append(int(elem))
            except ValueError:
                try:
                    instance.append(float(elem))
                except ValueError:
                    instance.append(elem)
        dataset.append(instance)
        line = f.readline()
    f.close()
    # handle missing values by taking average
    averages = []
    for i in range(len(dataset[0])):
        averages.append(compute_average(dataset, i))
    
    for instance in dataset:
        for i in range(len(instance)):
            if instance[i] == "NA":
                instance[i] = averages[i]
            
    return dataset

def compute_average(data, col):
    count = 0
    sum = 0
    for instance in data:
        if type(instance[col]) == str:
            continue
        sum += instance[col]
        count += 1
    if count == 0:
        return 0
    average = sum / count
    return average


'''
------------------------------------------------------------------------
STEP 1
------------------------------------------------------------------------
'''

def create_frequency_diagram(data, column, name):
    plt.figure()
    # data processing: Create unique list of elements and get their frequencies
    xs, ys = process_data(data, column)
    # create plot
    plt.bar(xs, ys, 0.45)
    name_string = "Total Number of Cars by " + name
    plt.title(name_string)
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.grid(True)
    plt.xticks(xs)
    pdf_name = "step-1-" + name + ".pdf"
    plt.savefig(pdf_name)
    #plt.show()




'''
------------------------------------------------------------------------
STEP 2
------------------------------------------------------------------------
'''

def create_pie_chart(data, column, name):
    plt.figure(figsize=(8,8))
    # data processing: Create unique list of elements and get their frequencies
    xs, ys = process_data(data, column)
    # create plot
    plt.pie(ys, labels=xs, autopct="%1.1f%%")
    name_string = "Total Number of Cars by " + name
    plt.title(name_string)
    pdf_name = "step-2-" + name + ".pdf"
    plt.savefig(pdf_name)
    #plt.show()



'''
------------------------------------------------------------------------
STEP 3
------------------------------------------------------------------------
'''

def create_dot_chart(data, column, name):
    plt.figure()
    # data processing: Create list of all elements, create dummy y-values\
    xs = []
    for row in data:
        xs.append(row[column])
    xs.sort()
    ys = [1] * len(xs)
    # create plot
    plt.plot(xs, ys, "b.", alpha=0.2, markersize = 16)
    plt.gca().get_yaxis().set_visible(False)
    name_string = name + " of all Cars"
    plt.title(name_string)
    plt.xlabel(name)
    pdf_name = "step-3-" + name + ".pdf"
    plt.savefig(pdf_name)
    #plt.show()



'''
------------------------------------------------------------------------
STEP 4
------------------------------------------------------------------------
'''

def create_discrete_graph_1(data):
    bins = []
    # data processing
    for i in range(10):
        bins.append(0)
    for instance in data:
        mpg = instance[0]
        if mpg <= 13:
            bins[0] += 1
        elif mpg <= 14:
            bins[1] += 1
        elif mpg <= 16:
            bins[2] += 1
        elif mpg <= 19:
            bins[3] += 1
        elif mpg <= 23:
            bins[4] += 1
        elif mpg <= 26:
            bins[5] += 1
        elif mpg <= 30:
            bins[6] += 1
        elif mpg <= 36:
            bins[7] += 1
        elif mpg <= 44:
            bins[8] += 1
        else:
            bins[9] += 1
    xs = range(10)
    # create plot
    plt.figure()
    plt.bar(xs, bins, 0.45)
    plt.grid(True)
    plt.xticks(xs, ["<= 13", "14", "15-16", "17-19", "20-23", "24-26", "27-30", "31-36", "37-44", ">= 45"])
    plt.title("Total Number of Cars by US Department of Energy Ratings of MPG")
    plt.xlabel("MPG")
    plt.ylabel("Count")
    plt.savefig("step-4-approach-1.pdf")
    #plt.show()


def create_discrete_graph_2(data):
    plt.figure()
    # data processing
    xs = []
    for instance in data:
        xs.append(instance[0])
    xs.sort()
    min = xs[0]
    max = xs[-1]
    bin_width = (max - min) / 5
    cutoffs = []
    for i in range(5):
        cutoffs.append(min + (bin_width * (i + 1)))
    bins = [0, 0, 0, 0, 0]
    for x in xs:
        for i in range(5):
            if x <= cutoffs[i]:
                bins[i] += 1
                break
    # create plot
    xrng = range(5)
    plt.bar(xrng, bins, 0.45)
    labels_i = [str(round(min))]
    for val in cutoffs:
        labels_i.append(str(round(val)))
    labels = []
    for i in range(5):
        label = labels_i[i] + "-" + labels_i[i + 1]
        labels.append(label)
    plt.xticks(xrng, labels=labels)
    plt.title("Total Number of Cars by Equal Width Rankings of MPG")
    plt.xlabel("MPG")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig("step-4-approach-2.pdf")
    #plt.show()
    


'''
------------------------------------------------------------------------
STEP 5
------------------------------------------------------------------------
'''

def create_histogram(data, column, name):
    plt.figure()
    # data processing
    xs = []
    for instance in data:
        xs.append(instance[column])
    xs.sort()
    # create plot
    plt.hist(xs, bins=10, alpha=0.75, color="b", ec="black")
    name_string = "Distribution of " + name + " Values"
    plt.title(name_string)
    plt.ylabel("Count")
    plt.xlabel(name)
    pdf_name = "step-5-" + name + ".pdf"
    plt.savefig(pdf_name)
    #plt.show()



'''
------------------------------------------------------------------------
STEP 6
------------------------------------------------------------------------
'''

def create_scatter_plot(data, column, name):
    plt.figure()
    # data processing
    xs = []
    ys = []
    for instance in data:
        xs.append(instance[column])
        ys.append(instance[0])
    # create plot
    plt.plot(xs, ys, "b.")
    name_string = name + " vs MPG"
    plt.title(name_string)
    plt.xlabel(name)
    plt.ylabel("MPG")
    plt.grid(True)
    pdf_name = "step-6-" + name + ".pdf"
    plt.savefig(pdf_name)
    #plt.show()



'''
------------------------------------------------------------------------
STEP 7
------------------------------------------------------------------------
'''

def create_linear_regression(data, column, name, column2, name2):
    plt.figure()
 # data processing
    xs = []
    ys = []
    for instance in data:
        xs.append(instance[column])
        ys.append(instance[column2])
    x_mean = (sum(xs))/(len(xs))
    y_mean = (sum(ys))/(len(ys))
    n = len(xs)
    if len(ys) != n:
        print("Something went wrong: Data arrays are different sizes!")
        return False
    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += ((xs[i] - x_mean)*(ys[i] - y_mean))
        denominator += ((xs[i] - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - (slope * x_mean)
    # Generate points for linear regression line
    xrng = np.arange(0, max(xs)+1)
    yrng = [((slope * x) + intercept) for x in xrng]
    # Calculate correlation coefficient r
    sum_ys = 0
    for y in ys:
        sum_ys += ((y - y_mean)**2)
    r = (numerator) / np.sqrt(denominator * sum_ys)
    # Calculate Covariance
    cov = numerator / n
    # Create plot
    plt.plot(xs, ys, '.', color="b")
    plt.plot(xrng, yrng, '-', color="red")
    plt.gca().annotate("corr: %.2f, cov: %.2f" %(r, cov),
                        xy = (0.5, 0.95), xycoords = 'axes fraction', color = "r",
                        bbox=dict(boxstyle="round", fc="1", color="r"))
    name_string = name + " vs " + name2
    plt.title(name_string)
    plt.xlabel(name2)
    plt.ylabel(name)
    plt.grid(True)
    pdf_name = "step-7-" + name + "-vs-" + name2 + ".pdf"
    plt.savefig(pdf_name)
    #plt.show()

    

'''
------------------------------------------------------------------------
STEP 8
------------------------------------------------------------------------
'''

def create_boxplot(data):
    plt.figure()
    # data processing
    model_years = {}
    for instance in data:
        year = instance[6]
        mpg = instance[0]
        if year in model_years:
            model_years[year].append(mpg)
        else:
            model_years[year] = [mpg]
    # Create plot
    plt.boxplot([model_years[year] for year in model_years])
    plt.title("MPG by Model Year")
    plt.ylabel("MPG")
    plt.xlabel("Model Year")
    plt.grid(True)
    xrng = np.arange(1, len([year for year in model_years]) + 1) 
    plt.xticks(xrng, [year for year in model_years])
    plt.savefig("step-8-mpg-by-Model-Year.pdf")
    #plt.show()



'''
------------------------------------------------------------------------
STEP 9
------------------------------------------------------------------------
'''

def create_multiple_frequency_diagram(data):
    plt.figure()
    # data processing
    frequencies = [{},{},{}]
    # populate with 0's
    for i in range(10):
        for o in range(3):
            frequencies[o][i + 70] = 0
    for instance in data:
        origin_index = instance[7] - 1
        year = instance[6]
        frequencies[origin_index][year] += 1
    origin1 = []
    for year in frequencies[0]:
        origin1.append(frequencies[0][year])
    origin2 = []
    for year in frequencies[1]:
        origin2.append(frequencies[1][year])
    origin3 = []
    for year in frequencies[2]:
        origin3.append(frequencies[2][year])   
    # Create plot
    ax = plt.gca()
    x_ticks = range(1, 11)
    y_ticks = range(0, 45, 5)
    r1 = ax.bar(x_ticks, origin1, 0.25, color="g")
    x_ticks = [x + 0.25 for x in x_ticks]
    r2 = ax.bar(x_ticks, origin2, 0.25, color="b")
    x_ticks = [x + 0.25 for x in x_ticks]
    r3 = ax.bar(x_ticks, origin3, 0.25, color="r")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(["70", "71", "72", "73", "74", "75", "76", "77", "78", "79"])
    ax.legend((r1[0], r2[0], r3[0]), ("US", "Europe", "Japan"), loc=2)
    plt.title("Total Number of Cars by Year and Country of Origin")
    plt.xlabel("Model Year")
    plt.ylabel("Count")
    plt.savefig("step-9-origin-and-mpg.pdf")
    #plt.show()



'''
------------------------------------------------------------------------
BONUS
------------------------------------------------------------------------
'''

def create_stem_plot(data):
    plt.figure()
    # data processing
    # Compute average mpg for each model year and total average
    averages_year = [0 for i in range(10)]
    frequencies_year = [0 for i in range(10)]
    average_total = 0
    frequency_total = 0
    for instance in data:
        mpg = instance[0]
        year = instance[6]
        averages_year[year - 70] += mpg
        average_total += mpg
        frequencies_year[year - 70] += 1
        frequency_total += 1
    for i in range(10):
        averages_year[i] = averages_year[i] / frequencies_year[i]
    average_total = average_total / frequency_total
    markerline, stemlines, baseline = plt.stem([i + 70 for i in range(10)], averages_year, '-.', bottom=average_total)
    plt.setp(baseline,  color='R', linewidth=2)
    plt.gca().annotate("Total Average: %.2f" %average_total, xy=(0.65, 0.44), 
                        xycoords="axes fraction", color="r",
                        bbox=dict(boxstyle="round", fc="1", color="r") )
    plt.title("Average MPG by Model Year vs Total Average MPG")
    plt.xlabel("Model Year")
    plt.ylabel("MPG (Average)")
    plt.xticks([i + 70 for i in range(10)])
    plt.savefig("bonus-mpg-average-by-Model-Year.pdf")
    #plt.show()


'''
------------------------------------------------------------------------
MAIN LOGIC
------------------------------------------------------------------------
'''

def main():
    auto_data = convert_dataset("auto-data-nodups.txt")
    # Step 1
    create_frequency_diagram(auto_data, 1, "Cylinders")
    create_frequency_diagram(auto_data, 6, "Model Year")
    create_frequency_diagram(auto_data, 7, "Origin")
    # Step 2
    create_pie_chart(auto_data, 1, "Cylinders")
    create_pie_chart(auto_data, 6, "Model Year")
    create_pie_chart(auto_data, 7, "Origin")
    # Step 3
    create_dot_chart(auto_data, 0, "MPG")
    create_dot_chart(auto_data, 2, "Displacement")
    create_dot_chart(auto_data, 3, "Horsepower")
    create_dot_chart(auto_data, 4, "Weight")
    create_dot_chart(auto_data, 5, "Acceleration")
    create_dot_chart(auto_data, 9, "MSRP")
    # Step 4
    create_discrete_graph_1(auto_data)
    create_discrete_graph_2(auto_data)
    # Step 5
    create_histogram(auto_data, 0, "MPG")
    create_histogram(auto_data, 2, "Displacement")
    create_histogram(auto_data, 3, "Horsepower")
    create_histogram(auto_data, 4, "Weight")
    create_histogram(auto_data, 5, "Acceleration")
    create_histogram(auto_data, 9, "MSRP")
    # Step 6
    create_scatter_plot(auto_data, 2, "Displacement")
    create_scatter_plot(auto_data, 3, "Horsepower")
    create_scatter_plot(auto_data, 4, "Weight")
    create_scatter_plot(auto_data, 5, "Acceleration")
    create_scatter_plot(auto_data, 9, "MSRP")
    # Step 7
    create_linear_regression(auto_data, 2, "Displacement", 0, "MPG")
    create_linear_regression(auto_data, 3, "Horsepower", 0, "MPG")
    create_linear_regression(auto_data, 4, "Weight", 0, "MPG")
    create_linear_regression(auto_data, 5, "Acceleration", 0, "MPG")
    create_linear_regression(auto_data, 9, "MSRP", 0, "MPG")
    create_linear_regression(auto_data, 2, "Displacement", 4, "Weight")
    # Step 8
    create_boxplot(auto_data)
    # Step 9
    create_multiple_frequency_diagram(auto_data)
    # Bonus
    create_stem_plot(auto_data)




main()