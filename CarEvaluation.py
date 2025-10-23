# This program classifies cars using two sets of hand-written rules, then checks how well the rules work.

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
    # Mostly looking at buying price and safety here
    if car['buying'] in ['vhigh', 'high'] and car['safety'] == 'low':
        return 'unacc'  # High price but low safety means unacceptable
    if car['buying'] in ['med', 'low'] and car['safety'] in ['med', 'high']:
        return 'acc'    # Mid or low price and good safety means acceptable
    if car['persons'] == 'more' and car['safety'] == 'high':
        return 'good'   # Can hold more people and is safe, so good
    if car['maint'] in ['low', 'med'] and car['lug_boot'] == 'big':
        return 'vgood'  # Low or medium maintenance with big boot is very good
    return 'acc'        # If none of the above, just say acceptable


def rule_based_classifier_2(car):
    # A different rule approach
    # Focus here is more on maintenance and doors
    if car['maint'] in ['vhigh', 'high'] and car['safety'] == 'low':
        return 'unacc'  # High maintenance and low safety is unacceptable
    if car['doors'] in ['2', '3'] and car['persons'] == '2':
        return 'unacc'  # Small door count with only 2 people isn't good
    if car['maint'] in ['low', 'med'] and car['lug_boot'] == 'big' and car['safety'] == 'high':
        return 'vgood'  # Low maintenance, big boot, and high safety is very good
    if car['buying'] in ['med', 'low'] and car['persons'] in ['4', 'more']:
        return 'good'   # Medium or low price and can hold 4+ are good cars
    return 'acc'        # Default to acceptable


def evaluate_classifier(cars, actual_labels, classifier):
    # Compares what the rules predicted to the real answers and check how many are right
    correct = 0
    total = len(cars)
    for i, car in enumerate(cars):
        prediction = classifier(car)
        actual = actual_labels[i]
        print(f"Car: {car}, Predicted: {prediction}, Actual: {actual}")
        if prediction == actual:
            correct += 1
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%\n")
    return accuracy


if __name__ == "__main__":
    # Load the data, run both rule sets on it,
    # This prints the results and accuracy for each one
    data_path = 'car.data'
    cars, labels = load_car_data(data_path)

    print("Algorithm 1 results:")
    accuracy1 = evaluate_classifier(cars, labels, rule_based_classifier_1)

    print("Algorithm 2 results:")
    accuracy2 = evaluate_classifier(cars, labels, rule_based_classifier_2)

    print(f"Final accuracy for Algorithm 1: {accuracy1:.2f}%")
    print(f"Final accuracy for Algorithm 2: {accuracy2:.2f}%")
