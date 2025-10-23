def load_car_data(filepath):
    # Reads car data from a file and return the features and actual labels
    cars = []
    labels = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            car = {
                'buying': parts[0],
                'maint': parts[1],
                'doors': parts[2],
                'persons': parts[3],
                'lug_boot': parts[4],
                'safety': parts[5]
            }
            cars.append(car)
            labels.append(parts[6])
    return cars, labels

if __name__ == "__main__":
    data_path = 'car.data'
    cars, labels = load_car_data(data_path)
    print("Loaded car data:")
    for i in range(5):
        print(cars[i], "->", labels[i])
