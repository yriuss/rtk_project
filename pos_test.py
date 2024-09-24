#!/usr/bin/env python3
import csv
import time
from pyubx2 import UBXReader
import serial

def main():
    """Main function to read GNSS data and save latitude and longitude to a CSV file."""
    try:
        ser = serial.Serial('COM10', 38400, timeout=5)
        print("Serial port opened successfully.")
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    reader = UBXReader(ser)
    
    try:
        data_buffer = []  # Buffer to hold all data until program ends
        
        print("Starting GNSS data collection. Press Ctrl+C to stop.")
        
        while True:
            try:
                raw_data, parsed_data = reader.read()
                
                if parsed_data.identity == "NAV-PVT":
                    lat = parsed_data.lat
                    lon = parsed_data.lon
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    
                    # Append data to buffer
                    data_buffer.append([timestamp, lat, lon])
                    
                    # Print to console for debugging
                    print(f"Timestamp: {timestamp}, Latitude: {lat}, Longitude: {lon}")
                    
            except Exception as e:
                print(f"Error while reading GNSS data: {e}")

            # Sleep for a short duration to avoid overloading the CPU
            #time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Data collection stopped. Saving to CSV...")
        
        # Write buffered data to CSV when program stops
        try:
            with open('gnss_positions.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Latitude', 'Longitude'])  # Write header
                writer.writerows(data_buffer)
                print(f"Saved {len(data_buffer)} records to CSV.")
        except Exception as e:
            print(f"Error saving to CSV file: {e}")
    
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == '__main__':
    main()
