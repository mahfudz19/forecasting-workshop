import unittest
from src.app.core import YourMainClass, your_main_function

class TestCore(unittest.TestCase):

    def setUp(self):
        self.instance = YourMainClass()

    def test_your_main_function(self):
        result = your_main_function()
        self.assertEqual(result, expected_value)

    def test_instance_method(self):
        result = self.instance.your_instance_method()
        self.assertEqual(result, expected_value)

if __name__ == '__main__':
    unittest.main()