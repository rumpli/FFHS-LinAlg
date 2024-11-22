def get_unit_matrix(size: int) -> list[list[int]]:
    """Generates an identity matrix of the given size.

    Args:
        size (int): The size of the identity matrix.

    Returns:
        list[list[int]]: The identity matrix.
    """
    identity_matrix = []
    for row_index in range(size):
        row = []
        for col_index in range(size):
            if row_index == col_index:
                row.append(1)
            else:
                row.append(0)
        identity_matrix.append(row)
    return identity_matrix


def extract_data_bits(encoded_word: str) -> str:
    """Extracts the data bits from an encoded word.

    Args:
        encoded_word (str): The encoded word.

    Returns:
        str: The extracted data bits.
    """
    data_bits = []
    parity_bit_index = 0
    for bit_position in range(1, len(encoded_word) + 1):
        if bit_position == 2 ** parity_bit_index: # Skip parity bits
            parity_bit_index += 1
        else:
            data_bits.append(encoded_word[-bit_position])
    return ''.join(data_bits[::-1])


class HammingCode:
    def __init__(self, parity_bits: int):
        """Initializes the HammingCode class with the given number of parity bits.

        Args:
            parity_bits (int): The number of parity bits.

        Raises:
            ValueError: If the number of parity bits is less than 2.
        """
        if parity_bits < 2:
            raise ValueError("Number of parity bits must be at least 2.")
        self.parity_bits = parity_bits
        self.codeword_length = (2 ** self.parity_bits) - 1
        self.data_length = self.codeword_length - self.parity_bits
        self.generator_matrix = self.get_generator_matrix()
        self.check_matrix = self.get_check_matrix()

        print(f"Constructed ({self.codeword_length}, {self.data_length}) Hamming code with {self.parity_bits} parity bits.")

    def pos_redundant_bits(self, data: str) -> str:
        """Positions the redundant bits in the data word.

        Args:
            data (str): The data word.

        Returns:
            str: The data word with positioned redundant bits.
        """
        parity_bit_index = 0
        data_bit_index = 1
        data_length = len(data)
        result = ''

        for bit_position in range(1, data_length + self.parity_bits + 1):
            if bit_position == 2 ** parity_bit_index:
                result += '0'
                parity_bit_index += 1
            else:
                result += str(data[-1 * data_bit_index])
                data_bit_index += 1
        return result[::-1]

    def calc_parity_bits(self, arr: str) -> str:
        """Calculates the parity bits for the given array.

        Args:
            arr (str): The array with positioned redundant bits.

        Returns:
            str: The array with calculated parity bits.
        """
        array_length = len(arr)
        arr = list(arr)

        for parity_bit_index in range(self.parity_bits):
            parity_value = 0
            for bit_position in range(1, array_length + 1):
                if bit_position & (2 ** parity_bit_index) == (2 ** parity_bit_index):
                    parity_value ^= int(arr[-bit_position])
            arr[-(2 ** parity_bit_index)] = str(parity_value)  # Set the parity bit

        return ''.join(arr)

    def detect_error(self, arr: str) -> int:
        """Detects the error position in the encoded word.

        Args:
            arr (str): The encoded word.

        Returns:
            int: The position of the error (0 if no error).
        """
        arr = [int(bit) for bit in arr[::-1]]
        syndrome = matrix_multiply(self.check_matrix, arr) # Calculate syndrome

        return int(''.join(map(str, syndrome[::-1])), 2)

    def encode(self, data: str) -> str:
        """Encodes the data word using Hamming code.

        Args:
            data (str): The data word.

        Returns:
            str: The encoded word.
        """
        # Position the redundant bits in the data word
        positioned_bits = self.pos_redundant_bits(data)

        # Calculate the parity bits for the positioned bits
        encoded_word = self.calc_parity_bits(positioned_bits)

        # Return the encoded word
        return encoded_word

    def decode(self, encoded_word: str, error: bool = False) -> tuple[str, str, int, bool]:
        """Decodes the encoded word and detects/corrects any errors.

        Args:
            encoded_word (str): The encoded word.
            error (bool, optional): Flag to indicate if an error was detected. Defaults to False.

        Returns:
            tuple[str, str, int, bool]: The decoded word, received data word, error position, and error flag.
        """
        correction = self.detect_error(encoded_word)
        if correction != 0:
            error = True
            encoded_word = list(encoded_word)
            correction = len(encoded_word) - correction
            print(f"Error detected at position {correction}. Correcting bit...")
            encoded_word[correction] = '1' if encoded_word[correction] == '0' else '0'
            encoded_word = ''.join(encoded_word)

        return encoded_word, extract_data_bits(encoded_word), correction, error

    def check(self, codeword: str) -> bool:
        """Checks if the codeword is a valid Hamming codeword.

        Args:
            codeword (str): The codeword to check.

        Returns:
            bool: True if the codeword is valid, False otherwise.
        """
        result = matrix_multiply(self.check_matrix, [int(bit) for bit in reversed(codeword)])

        # Return True if all elements of the result are zero (valid codeword)
        return all(x == 0 for x in result)

    def get_generator_matrix(self) -> list[list[int]]:
        """Generates the generator matrix for the Hamming code.

        Returns:
            list[list[int]]: The generator matrix.
        """
        identity_matrix = get_unit_matrix(self.data_length)
        parity_check_matrix = self.get_check_matrix()
        generator_matrix = []

        for data_bit_index in range(self.data_length):
            row = identity_matrix[data_bit_index]
            for parity_bit_index in range(self.parity_bits):
                row.append(parity_check_matrix[parity_bit_index][data_bit_index])
            generator_matrix.append(row)

        return generator_matrix

    def get_check_matrix(self) -> list[list[int]]:
        """Generates the check matrix for the Hamming code.

        Returns:
            list[list[int]]: The check matrix.
        """
        parity_check_matrix = []

        for parity_bit_index in range(self.parity_bits):
            row = []
            for bit_position in range(1, self.codeword_length + 1):
                if bit_position & (1 << parity_bit_index):
                    row.append(1)
                else:
                    row.append(0)
            parity_check_matrix.append(row)

        return parity_check_matrix


def matrix_multiply(matrix: list[list[int]], vector: list[int]) -> list[int]:
    """Performs matrix multiplication of a matrix and a vector.

    Args:
        matrix (list[list[int]]): The matrix.
        vector (list[int]): The vector.

    Returns:
        list[int]: The result of the matrix multiplication.
    """
    result = []
    for row in matrix:
        row_sum = sum(x * y for x, y in zip(row, vector)) % 2 # Perform modulo 2 addition
        result.append(row_sum)
    return result


def ask_user_for_bits() -> int:
    """Prompts the user to enter the number of redundant bits for the Hamming code.

    Returns:
        int: The number of redundant bits.
    """
    print("This program will encode a data word using Hamming code.")
    print("You can choose to introduce an error in the encoded word.")
    print("The program will then decode the word and detect the error.")
    print("The error will be corrected if possible.")
    print()
    print("Please define your hamming code by entering the number of redundant bits.")
    print("The number of data bits will be calculated automatically.")
    print()
    while True:
        try:
            parity_bits = int(input("Enter the number of redundant bits (m): "))
            if parity_bits > 0:
                break
            print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    return parity_bits


def ask_user_for_data_word(hamming: HammingCode) -> str:
    """Prompts the user to enter a data word of the appropriate length.

    Args:
        hamming (HammingCode): The HammingCode instance.

    Returns:
        str: The data word entered by the user.
    """
    while True:
        data_word = input(f"Enter a data word with {hamming.data_length} bits: ").strip()
        if len(data_word) == hamming.data_length and all(bit in "01" for bit in data_word):
            break
        print(f"Invalid input. Please enter a {hamming.data_length}-bit binary number.")
    return data_word


def ask_user_for_error(hamming: HammingCode, encoded_word: str) -> str:
    """Prompts the user to introduce an error in the encoded word.

    Args:
        hamming (HammingCode): The HammingCode instance.
        encoded_word (str): The encoded word.

    Returns:
        str: The encoded word with the introduced error (if any).
    """
    while True:
        introduce_error = input("Do you want to introduce an error? (yes/no): ").strip().lower()
        if introduce_error in ["yes", "no"]:
            break
        print("Invalid input. Please type 'yes' or 'no'.")
    if introduce_error == "yes":
        while True:
            try:
                print(f"Introduce error in encoded word: {encoded_word}")
                error_position = int(input(f"Enter a position (0 to {hamming.codeword_length - 1}) to introduce an error: "))
                if 0 <= error_position < hamming.codeword_length:
                    encoded_word = list(encoded_word)
                    encoded_word[error_position] = str(int(encoded_word[error_position]) ^ 1)
                    encoded_word = ''.join(encoded_word)
                    print(f"Encoded word with error at position {error_position}: ", encoded_word)
                    break
                else:
                    print(f"Invalid position. Please choose a value between 0 and {hamming.codeword_length - 1}.")
            except ValueError:
                print("Please enter a valid integer.")
    else:
        print("No error introduced.")
    return encoded_word


def main():
    """Main function to run the Hamming code program."""
    position = 25
    try:
        hamming = HammingCode(ask_user_for_bits())
        data_word = ask_user_for_data_word(hamming)
        og_encoded_word = encoded_word = hamming.encode(data_word)
        encoded_word = ask_user_for_error(hamming, encoded_word)
        check = hamming.check(encoded_word)
        decoded_word, rec_data_word, error_position, error = hamming.decode(encoded_word)
        print("------------------------------------------------")
        print("Check matrix:")
        for row in hamming.check_matrix:
            print(row)
        print("Generator matrix:")
        for row in hamming.generator_matrix:
            print(row)
        print("------------------------------------------------")
        print(f"{'Data word:':<{position}} {data_word}")
        print(f"{'Received data word:':<{position}} {rec_data_word}")
        print("------------------------------------------------")
        print(f"{'Sent encoded word:':<{position}} {og_encoded_word}")
        print(f"{'Received encoded word:':<{position}} {encoded_word}")
        print(f"{'Encoded word is correct:':<{position}} {check}")
        print("------------------------------------------------")
        print(f"{'Encoded word:':<{position}} {decoded_word}")
        if error:
            print(f"{'':<{position + error_position}} ^")
            print(f"{'Error at position:':<{position+error_position}} {error_position}")
        else:
            print("No error detected.")
        print("------------------------------------------------")
    except ValueError as e:
        print(e)
        return

if __name__ == '__main__':
    main()