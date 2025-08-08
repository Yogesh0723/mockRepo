import pandas as pd
import json
from faker import Faker
import random
import argparse
from pathlib import Path
# python AddressDataGenerator.py --samples 10000 --format both --output_dir ./data

class AddressDataGenerator:
    def __init__(self):
        self.fake = Faker(['en_GB'])  # Multiple locales for variety
        
        # Define address components
        self.flat_prefixes = ['FLAT', 'APT', 'APARTMENT', 'UNIT', 'SUITE']
        self.house_name_suffixes = ['HOUSE', 'COTTAGE', 'MANOR', 'LODGE', 'VILLA', 'COURT']
        self.street_types = ['STREET', 'ST', 'ROAD', 'RD', 'AVENUE', 'AVE', 'LANE', 'LN', 
                           'DRIVE', 'DR', 'CLOSE', 'CL', 'PLACE', 'PL', 'CRESCENT', 'CR']
        
        # BIO tagging scheme
        self.labels = {
            'O': 'O',  # Outside
            'B-FLAT': 'B-FLAT',  # Beginning of flat number
            'I-FLAT': 'I-FLAT',  # Inside flat number
            'B-HOUSE_NUM': 'B-HOUSE_NUM',  # Beginning of house number
            'I-HOUSE_NUM': 'I-HOUSE_NUM',  # Inside house number
            'B-HOUSE_NAME': 'B-HOUSE_NAME',  # Beginning of house name
            'I-HOUSE_NAME': 'I-HOUSE_NAME',  # Inside house name
            'B-STREET': 'B-STREET',  # Beginning of street
            'I-STREET': 'I-STREET'   # Inside street
        }
    
    def generate_flat_number(self):
        """Generate flat/apartment number"""
        return f"{random.choice(self.flat_prefixes)} {random.randint(1, 999)}"
    
    def generate_house_number(self):
        """Generate house number (can be numeric or alphanumeric)"""
        if random.random() < 0.8:  # 80% numeric
            return str(random.randint(1, 9999))
        else:  # 20% alphanumeric
            return f"{random.randint(1, 999)}{random.choice(['A', 'B', 'C'])}"
    
    def generate_house_name(self):
        """Generate house name"""
        if random.random() < 0.3:  # 30% chance of traditional house name
            return f"{self.fake.last_name().upper()} {random.choice(self.house_name_suffixes)}"
        else:  # 70% chance of descriptive name
            adjectives = ['THE', 'OLD', 'NEW', 'ROYAL', 'GRAND', 'LITTLE']
            nouns = ['OAKS', 'WILLOWS', 'GARDENS', 'MEADOWS', 'HEIGHTS', 'PARK']
            return f"{random.choice(adjectives)} {random.choice(nouns)}"
    
    def generate_street_name(self):
        """Generate street name"""
        street_name = self.fake.street_name().upper().replace(' STREET', '').replace(' ROAD', '').replace(' AVENUE', '')
        street_type = random.choice(self.street_types)
        return f"{street_name} {street_type}"
    
    def create_bio_tags(self, tokens, components):
        tags = ['O'] * len(tokens)
        used_indices = set()

        def assign_tags(comp_type, comp_value):
            if not comp_value:
                return

            comp_tokens = comp_value.split()
            for i in range(len(tokens) - len(comp_tokens) + 1):
                # Ensure no overlap
                if any(idx in used_indices for idx in range(i, i + len(comp_tokens))):
                    continue

                window = tokens[i:i + len(comp_tokens)]

                if window == comp_tokens:
                    tags[i] = f'B-{comp_type.upper()}'
                    used_indices.update([i])
                    for j in range(1, len(comp_tokens)):
                        tags[i + j] = f'I-{comp_type.upper()}'
                        used_indices.update([i + j])
                    break

            # Assign tags in priority order
        assign_tags('flat', components.get('flat'))
        assign_tags('house_num', components.get('house_num'))
        assign_tags('house_name', components.get('house_name'))
        assign_tags('street', components.get('street'))

        return tags
    
    def generate_address_sample(self):
        """Generate a single address sample with labels"""
        components = {
            'flat': None,
            'house_num': None,
            'house_name': None,
            'street': None
        }
        
        # Randomly decide which components to include
        include_flat = random.random() < 0.3  # 30% chance
        include_house_num = random.random() < 0.7  # 70% chance
        include_house_name = random.random() < 0.4  # 40% chance
        include_street = True  # Always include street
        
        address_parts = []
        
        if include_flat:
            flat_num = self.generate_flat_number()
            components['flat'] = flat_num
            address_parts.append(flat_num)
        
        if include_house_num:
            house_num = self.generate_house_number()
            components['house_num'] = house_num
            address_parts.append(house_num)
        
        if include_house_name:
            house_name = self.generate_house_name()
            components['house_name'] = house_name
            address_parts.append(house_name)
        
        if include_street:
            street = self.generate_street_name()
            components['street'] = street
            address_parts.append(street)
        
        # Join address parts
        address = ' '.join(address_parts)
        
        # Tokenize
        tokens = address.split()
        
        # Create BIO tags
        bio_tags = self.create_bio_tags(tokens, components)
        
        return {
            'address': address,
            'tokens': tokens,
            'bio_tags': bio_tags,
            'flat_number': components['flat'] or '',
            'house_number': components['house_num'] or '',
            'house_name': components['house_name'] or '',
            'street': components['street'] or ''
        }
    
    def generate_dataset(self, num_samples=10000):
        """Generate a dataset of addresses with NER labels"""
        data = []
        
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples} samples...")
            
            sample = self.generate_address_sample()
            data.append(sample)
        
        print(f"Generated {num_samples} samples successfully!")
        return data
    
    def save_to_csv(self, data, filename='address_dataset.csv'):
        """Save dataset to CSV format"""
        csv_data = []
        
        for sample in data:
            # Create rows for token-level data
            for i, (token, tag) in enumerate(zip(sample['tokens'], sample['bio_tags'])):
                csv_data.append({
                    'address': sample['address'],
                    'token_id': i,
                    'token': token,
                    'bio_tag': tag,
                    'flat_number': sample['flat_number'],
                    'house_number': sample['house_number'],
                    'house_name': sample['house_name'],
                    'street': sample['street']
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"CSV Shape: {df.shape}")
        return df
    
    def save_to_json(self, data, filename='address_dataset.json'):
        """Save dataset to JSON format"""
        # Convert to JSON-serializable format
        json_data = []
        
        for sample in data:
            json_data.append({
                'address': sample['address'],
                'tokens': sample['tokens'],
                'bio_tags': sample['bio_tags'],
                'entities': {
                    'flat_number': sample['flat_number'],
                    'house_number': sample['house_number'],
                    'house_name': sample['house_name'],
                    'street': sample['street']
                }
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
        print(f"JSON entries: {len(json_data)}")
        return json_data
    
    def load_from_csv(self, filename):
        """Load dataset from CSV"""
        df = pd.read_csv(filename)
        
        # Group by address to reconstruct samples
        grouped = df.groupby('address')
        data = []
        
        for address, group in grouped:
            sample = {
                'address': address,
                'tokens': group['token'].tolist(),
                'bio_tags': group['bio_tag'].tolist(),
                'flat_number': group['flat_number'].iloc[0],
                'house_number': group['house_number'].iloc[0],
                'house_name': group['house_name'].iloc[0],
                'street': group['street'].iloc[0]
            }
            data.append(sample)
        
        return data
    
    def load_from_json(self, filename):
        """Load dataset from JSON"""
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        data = []
        for item in json_data:
            sample = {
                'address': item['address'],
                'tokens': item['tokens'],
                'bio_tags': item['bio_tags'],
                'flat_number': item['entities']['flat_number'],
                'house_number': item['entities']['house_number'],
                'house_name': item['entities']['house_name'],
                'street': item['entities']['street']
            }
            data.append(sample)
        
        return data
    
    def print_sample_data(self, data, num_samples=5):
        """Print sample data for inspection"""
        print(f"\nSample data (first {num_samples} entries):")
        print("=" * 80)
        
        for i, sample in enumerate(data[:num_samples]):
            print(f"\nSample {i+1}:")
            print(f"Address: {sample['address']}")
            print(f"Tokens: {sample['tokens']}")
            print(f"BIO Tags: {sample['bio_tags']}")
            print(f"Flat Number: '{sample['flat_number']}'")
            print(f"House Number: '{sample['house_number']}'")
            print(f"House Name: '{sample['house_name']}'")
            print(f"Street: '{sample['street']}'")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Generate address NER training data')
    parser.add_argument('--samples', type=int, default=5000, 
                        help='Number of samples to generate (default: 5000)')
    parser.add_argument('--format', choices=['csv', 'json', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--filename', type=str, default='address_dataset',
                        help='Base filename (default: address_dataset)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = AddressDataGenerator()
    
    # Generate data
    print(f"Generating {args.samples} address samples...")
    data = generator.generate_dataset(args.samples)
    
    # Print sample data
    generator.print_sample_data(data)
    
    # Save data
    if args.format in ['csv', 'both']:
        csv_path = output_dir / f"{args.filename}.csv"
        generator.save_to_csv(data, csv_path)
    
    if args.format in ['json', 'both']:
        json_path = output_dir / f"{args.filename}.json"
        generator.save_to_json(data, json_path)
    
    print(f"\nData generation completed!")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    main()