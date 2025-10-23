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


def rule_based_classifier_1(car):
    # My first set of rules to decide car classes
    if car['buying'] in ['vhigh', 'high'] and car['safety'] == 'low':
        return 'unacc'
    if car['buying'] in ['med', 'low'] and car['safety'] in ['med', 'high']:
        return 'acc'
    if car['persons'] == 'more' and car['safety'] == 'high':
        return 'good'
    if car['maint'] in ['low', 'med'] and car['lug_boot'] == 'big':
        return 'vgood'
    return 'acc'

def rule_based_classifier_2(car):
    # A different rule approach
    if car['maint'] in ['vhigh', 'high'] and car['safety'] == 'low':
        return 'unacc'
    if car['doors'] in ['2', '3'] and car['persons'] == '2':
        return 'unacc'
    if car['maint'] in ['low', 'med'] and car['lug_boot'] == 'big' and car['safety'] == 'high':
        return 'vgood'
    if car['buying'] in ['med', 'low'] and car['persons'] in ['4', 'more']:
        return 'good'
    return 'acc'

if __name__ == "__main__":
    data_path = 'car.data'
    cars, labels = load_car_data(data_path)

    print("Classifier 1 prediction:", rule_based_classifier_1(cars[0]))
    print("Classifier 2 prediction:", rule_based_classifier_2(cars[0]))

