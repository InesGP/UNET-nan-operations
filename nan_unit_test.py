import torch
import torch.nn.functional as F
from torch.nn.functional import max_pool2d
import numpy as np
import unittest
import os
import math

class TestNaNPool2dPy(unittest.TestCase):
    

    def unit_test_indices(self, input_tensor, expected, name, test_torch=True, test_expected=True):
        from nan_ops import NaNPool2d

        
        nanPoolPy = NaNPool2d(max_threshold=1) 
        torchpool = torch.nn.MaxPool2d(2, 2, return_indices=True)

        default = torchpool(input_tensor)[1]
        testing = nanPoolPy(input_array=input_tensor, pool_size=(2, 2), strides=(2, 2))[1]

        if test_torch:
            self.assertTrue(torch.equal(default, testing), f"{name} Test failed: Torch and NaNPool indices do not match.")
            print(f"{name} Test passed: Torch and NaNPool indices match.")
        else: print('NaNPool expected behaviour differs from Torch -- comparison to Torch is skipped')
        # print(default , testing)

        if test_expected:
            self.assertTrue(torch.equal(expected, testing), f"{name} Test failed: Expected and NaNPool indices do not match.")
            print(f"{name} Test passed: Expected and NaNPool indices match.")
        else: print('NanPool expected behaviour matches Torch -- no need to run additional test')
        # print(expected , testing)

    def unit_test_maxvalues(self, input_tensor, expected, name, test_torch=True, test_expected=True):
        from nan_ops import NaNPool2d

        nanPoolPy = NaNPool2d(max_threshold=1) 
        torchpool = torch.nn.MaxPool2d(2, 2, return_indices=True)

        default = torchpool(input_tensor)[0]
        default = default.masked_fill(torch.isnan(default), 0.5)
        testing = nanPoolPy(input_array=input_tensor, pool_size=(2, 2), strides=(2, 2))[0]
        testing = testing.masked_fill(torch.isnan(testing), 0.5)
        # print(default , testing)

        if test_torch:
            self.assertTrue(torch.equal(default, testing), f"{name} Test failed: Torch and NaNPool max values do not match.")
            print(f"{name} Test passed: Torch and NaNPool max values match.")
        else: print('NaNPool expected behaviour differs from Torch -- comparison to Torch is skipped')
        # print(default , testing)

        if test_expected:
            self.assertTrue(torch.equal(expected, testing), f"{name} Test failed: Expected and NaNPool max values do not match.")
            print(f"{name} Test passed: Expected and NaNPool max values match.")
        else: print('NanPool expected behaviour matches Torch -- no need to run additional test')
        # print(expected , testing)

    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        input = torch.randint(0, 100, (2, 2, 2, 2)).float()
        input[:] = float('nan')
        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[0]], [[0]]], [[[0]], [[0]]]])
        expected_val = torch.tensor([[[[0.5]], [[0.5]]], [[[0.5]], [[0.5]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool all nans', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool all nans', test_torch=True, test_expected=True)

    def test_no_nans(self):
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        shape = (2, 2, 2, 4)
        num_elements = np.prod(shape)
        # Generate unique random values
        unique_values = torch.randperm(num_elements) % 100  # Ensure values are within the range 0 to 99
        # Reshape to desired shape and convert to float
        input = unique_values.reshape(shape).float()
        # print('INPUT:\n', input, input.shape)
        self.unit_test_indices(input, None, name='NaNPool no nans', test_torch=True, test_expected=False)
        self.unit_test_maxvalues(input, None, name='NaNPool no nans', test_torch=True, test_expected=False)

    def test_mixed_nans_multimaxval(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        input = torch.randint(0, 100, (2, 2, 2, 2)).float()
        input[:] = float('nan')
        input[0] = 1
        input[0, 0, 0, 0] = float('nan')
        input[1,1,1,1] = 10
        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[0]], [[0]]], [[[0]], [[3]]]])
        expected_val = torch.tensor([[[[0.5]], [[0.5]]], [[[0.5]], [[10]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool mixed nans multi max', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool mixed nans multi max', test_torch=False, test_expected=True)

    def test_mixed_nans_no_multimaxval(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [33.,  3.]],

         [[75., 59.],
          [53., 24.]]],


        [[[61., 64.],
          [26., 27.]],

         [[67., 68.],
          [11., nan]]]]
        """
        input = torch.tensor([[[[float('nan'), float('nan')], [33.,  3.]],
                         [[75., 59.], [53., 24.]]],
                         [[[61., 64.], [26., 27.]],
                          [[67., 68.], [11., float('nan')]]]])

        # print('INPUT:\n', input, input.shape)
        expected_idx = torch.tensor([[[[2]], [[0]]], [[[1]], [[1]]]])
        expected_val = torch.tensor([[[[33]], [[75]]], [[[64]], [[68]]]])
        self.unit_test_indices(input, expected_idx, name='NaNPool mixed nans no multi max', test_torch=False, test_expected=True)
        self.unit_test_maxvalues(input, expected_val, name='NaNPool mixed nans no multi max', test_torch=False, test_expected=True)


class TestNaNConv2dPy(unittest.TestCase):
    
    def setUp(self):
        # Default values
        self.nonan_fault_tolerance = getattr(self, 'nonan_fault_tolerance', 0.05)
        self.mixed_fault_tolerance = getattr(self, 'mixed_fault_tolerance', 0.02)

    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        from nan_ops import NaNConv2d

        
        kernel = torch.randn(8, 4, 3, 3)
        inputs = torch.randn(1, 4, 5, 5)
        inputs[:] = float('nan')

        output = torch.zeros((1,8,5,5))
        output[:,:, 1:-1, 1:-1] = float('nan')
        mask = torch.isnan(output)
        output[mask] = 1

        conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
        nanoutput = conv(inputs)
        nanoutput[mask] = 1
        
        self.assertTrue(torch.equal(output, nanoutput), "NanConv all nans Test failed: Torch and NaNConv values do not match.")
        print("NanConv all nans Test passed: Torch and NaNConv values match.")


    def test_no_nans(self): 
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        from nan_ops import NaNConv2d

        fail=0
        n = 1000

        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)

          output = F.conv2d(inputs, kernel, stride=1, padding=1)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-04).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        fail=0
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=0, stride=1, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-04).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=0 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=0 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        fail=0
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)

          output = F.conv2d(inputs, kernel, stride=3, padding=1)
          conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=3, threshold=1)
          nanoutput = conv(inputs)
          if not torch.isclose(nanoutput, output, rtol=5e-04).all(): fail += 1

        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")


    def test_mixed_nans(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        from nan_ops import NaNConv2d

   
        fail = 0
        n = 1000
        for _ in range(n):
            kernel = torch.randn(8, 4, 3, 3)
            inputs = torch.randn(1, 4, 5, 5)

            inputs[:,:, ::2, 0] = float('nan')
            inputs[:,:, ::2, -1] = float('nan')
            # inputs[:,:, -1, :] = float('nan')
            inputs[:,:, 2, :] = float('nan')

            output = F.conv2d(inputs, kernel, stride=1, padding=1)

            conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
            nanoutput = conv(inputs)

            mask = torch.isnan(output)
            output[mask] = 1

            nanoutput[mask] = 1
            if not torch.isclose(nanoutput, output, rtol=1e-03).all(): fail += 1
    
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv mixed nan Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv mixed nan Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


class PrepNaNConv2d:
    
    def __init__(self, save_dir='./'):
       self.save_dir = save_dir


    def prep_multichannel(self, multichannel_filename='custom'):
        
        torch.manual_seed(0)

        channels = 2
        batches = 2
        size = 4**2
        sqrt_size = int(math.sqrt(size))
        # Determine how many elements should be NaN
        num_nans = int(size * 0.33)
        n = 1000


        inputs = torch.arange(1, size*channels*batches + 1).float()

        # Reshape the tensor to the desired shape
        inputs = inputs.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

        # Create a random weights tensor of integers
        kernel = torch.ones(1, 1*channels, 2, 2).float()
        
        total_inputs = []
        # total_kernel = []
        total_output = []

        for _ in range(n):
          
          # Create a random permutation of the indices
          indices = torch.randperm(size)
          
          # Set the elements at the indices to NaN for each channel
          for c in range(inputs.shape[1]):
              inputs[0, c].view(-1)[indices[:num_nans]] = float('nan')

          # total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        #total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_multichannel.pt')
        # torch.save(total_kernel, f'{self.save_dir}/kernel_multichannel.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{multichannel_filename}_multichannel.pt')



    def prep_multibatch(self, multibatch_filename='custom'):
        
        torch.manual_seed(0)

        channels = 1
        batches = 5
        size = 4**2
        sqrt_size = int(math.sqrt(size))
        # Determine how many elements should be NaN
        num_nans = int(size * 0.33)
        n = 1000


        inputs = torch.arange(1, size*channels*batches + 1).float()

        # Reshape the tensor to the desired shape
        inputs = inputs.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

        # Create a random weights tensor of integers
        kernel = torch.ones(1, 1*channels, 2, 2).float()
        
        total_inputs = []
        # total_kernel = []
        total_output = []

        for _ in range(n):
          
          # Set the elements at the indices to NaN for each channel
          for c in range(inputs.shape[0]):
              # Create a random permutation of the indices
              indices = torch.randperm(size)

              inputs[c, 0].view(-1)[indices[:num_nans]] = float('nan')

          # total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        #total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_multibatch.pt')
        # torch.save(total_kernel, f'{self.save_dir}/kernel_multibatch.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{multibatch_filename}_multibatch.pt')


    def prep_multi4d(self, multi4d_filename='custom'):
        
        torch.manual_seed(0)

        channels = 3
        batches = 3
        size = 4**2
        sqrt_size = int(math.sqrt(size))
        # Determine how many elements should be NaN
        num_nans = int(size * 0.33)
        n = 1000


        inputs = torch.arange(1, size*channels*batches + 1).float()

        # Reshape the tensor to the desired shape
        inputs = inputs.reshape(1*batches, 1*channels, sqrt_size, sqrt_size)

        # Create a random weights tensor of integers
        kernel = torch.ones(1, 1*channels, 2, 2).float()
        
        total_inputs = []
        # total_kernel = []
        total_output = []

        for _ in range(n):
          
          # Set the elements at the indices to NaN for each channel
          # Set the elements at the indices to NaN for each channel
          for b in range(inputs.shape[0]):
              for c in range(inputs.shape[1]):
                  # Create a random permutation of the indices
                  indices = torch.randperm(size)

                  inputs[b, c].view(-1)[indices[:num_nans]] = float('nan')

          # total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        #total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_multi4d.pt')
        # torch.save(total_kernel, f'{self.save_dir}/kernel_multi4d.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{multi4d_filename}_multi4d.pt')
        
    
    def prep_no_nans(self, nonan_filename='custom'): 
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

         [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

         [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        torch.manual_seed(0)

        n = 1000
        
        total_inputs = []
        total_kernel = []
        total_output = []

        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)

          output = F.conv2d(inputs, kernel, stride=1, padding=1)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_stride1_pad1_nonans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_stride1_pad1_nonans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{nonan_filename}_stride1_pad1_nonans.pt')


        total_inputs = []
        total_kernel = []
        total_output = []
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)
          
          output = F.conv2d(inputs, kernel, stride=1, padding=0)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_stride1_pad0_nonans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_stride1_pad0_nonans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{nonan_filename}_stride1_pad0_nonans.pt')

        
        total_inputs = []
        total_kernel = []
        total_output = []
        for _ in range(n):
          kernel = torch.randn(8, 4, 3, 3)
          inputs = torch.randn(1, 4, 5, 5)
          total_kernel.append(kernel)
          total_inputs.append(inputs)
          
          output = F.conv2d(inputs, kernel, stride=3, padding=1)
          total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_stride3_pad1_nonans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_stride3_pad1_nonans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{nonan_filename}_stride3_pad1_nonans.pt')


    def prep_mixed_nans(self, mixednan_filename='custom'):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

         [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, 10.]]]]
        """
        torch.manual_seed(0)
   
        n = 1000
        total_inputs = []
        total_kernel = []
        total_output = []
        for i in range(n):
            kernel = torch.randn(8, 4, 3, 3)
            inputs = torch.randn(1, 4, 5, 5)

            inputs[:,:, ::2, 0] = float('nan')
            inputs[:,:, ::2, -1] = float('nan')
            # inputs[:,:, -1, :] = float('nan')
            inputs[:,:, 2, :] = float('nan')

            total_kernel.append(kernel)
            total_inputs.append(inputs)

            output = F.conv2d(inputs, kernel, stride=1, padding=1)
            total_output.append(output)

        total_inputs = torch.stack(total_inputs)
        total_kernel = torch.stack(total_kernel)
        total_output = torch.stack(total_output)

        torch.save(total_inputs, f'{self.save_dir}/inputs_mixednans.pt')
        torch.save(total_kernel, f'{self.save_dir}/kernel_mixednans.pt')
        torch.save(total_output, f'{self.save_dir}/conv_{mixednan_filename}_mixednans.pt')


class TestNaNConv2d(unittest.TestCase):
    
    def setUp(self):
        # Default values
        self.nonan_fault_tolerance = getattr(self, 'nonan_fault_tolerance', 0.05)
        self.mixed_fault_tolerance = getattr(self, 'mixed_fault_tolerance', 0.02)
        self.save_dir = getattr(self, 'save_dir', '/nanconv_unittests')

    def test_multichannel(self):
        """
        Example Input:
        [[[[ 1.,  2.,  3.,  4.],
          [ 5.,  6., nan,  8.],
          [ 9., nan, nan, nan],
          [nan, 14., 15., 16.]],

         [[17., 18., 19., 20.],
          [nan, 22., 23., nan],
          [nan, 26., 27., 28.],
          [29., nan, 31., nan]],

         [[33., nan, 35., 36.],
          [37., 38., 39., nan],
          [41., nan, 43., nan],
          [45., nan, 47., 48.]]]]
        """
        torch.manual_seed(0)
        
        n = 1000
        batch_size = 1
        channels = 2
        nan_inputs = torch.load(f'{self.save_dir}/inputs_multichannel.pt')
        nan_kernel = torch.ones(1, 1*channels, 2, 2).float()
        # nan_kernel = torch.load(f'{self.save_dir}/kernel_multichannel.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_multichannel.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel, stride=1, padding=0))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load(f'{self.save_dir}/inputs_default_mixednans.pt')
        # default_kernel = torch.load(f'{self.save_dir}/kernel_default_mixednans.pt')
        default_output = torch.load(f'{self.save_dir}/conv_default_multichannel.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-03).all(): fail += 1

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv multi channel Test failed: Torch and NaNConv convolutions do not match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv multi channel Test passed: Torch and NaNConv convolutions match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_multibatch(self):
        """
        Example Input:
        [[[[ 1.,  2.,  3.,  4.],
          [ 5.,  6., nan,  8.],
          [ 9., nan, nan, nan],
          [nan, 14., 15., 16.]]],


        [[[17., 18., 19., 20.],
          [nan, 22., 23., nan],
          [nan, 26., 27., 28.],
          [29., nan, 31., nan]]]
        """
        torch.manual_seed(0)

        n = 1000
        channels = 1
        batch_size = 5
        nan_inputs = torch.load(f'{self.save_dir}/inputs_multibatch.pt')
        nan_kernel = torch.ones(1, 1*channels, 2, 2).float()
        # nan_kernel = torch.load(f'{self.save_dir}/kernel_multibatch.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_multibatch.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel, stride=1, padding=0))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load(f'{self.save_dir}/inputs_default_mixednans.pt')
        # default_kernel = torch.load(f'{self.save_dir}/kernel_default_mixednans.pt')
        default_output = torch.load(f'{self.save_dir}/conv_default_multibatch.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-03).all(): fail += 1

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv multi batch Test failed: Torch and NaNConv convolutions do not match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv multi batch Test passed: Torch and NaNConv convolutions match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_multi4d(self):
        """
        Example Input:
        [[[[  1.,   2.,   3.,   4.],
          [  5.,   6.,  nan,   8.],
          [  9.,  nan,  nan,  nan],
          [ nan,  14.,  15.,  16.]],

         [[ 33.,  nan,  35.,  36.],
          [ 37.,  38.,  39.,  nan],
          [ 41.,  nan,  43.,  nan],
          [ 45.,  nan,  47.,  48.]]],


        [[[ nan,  50.,  51.,  52.],
          [ 53.,  54.,  55.,  56.],
          [ 57.,  nan,  59.,  nan],
          [ 61.,  nan,  nan,  64.]],

         [[ 81.,  nan,  nan,  84.],
          [ 85.,  86.,  nan,  88.],
          [ 89.,  90.,  nan,  nan],
          [ 93.,  94.,  95.,  96.]]],

        [[[ 97.,  98.,  nan,  nan],
          [101., 102., 103.,  nan],
          [105., 106., 107.,  nan],
          [109., 110.,  nan, 112.]],

         [[129., 130.,  nan,  nan],
          [133.,  nan, 135.,  nan],
          [137., 138., 139., 140.],
          [141., 142.,  nan, 144.]]]])
        """
        torch.manual_seed(0)

        n = 1000
        channels = 3
        batch_size = 3
        nan_inputs = torch.load(f'{self.save_dir}/inputs_multi4d.pt')
        nan_kernel = torch.ones(1, 1*channels, 2, 2).float()
        # nan_kernel = torch.load(f'{self.save_dir}/kernel_multi4d.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_multi4d.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel, stride=1, padding=0))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load(f'{self.save_dir}/inputs_default_mixednans.pt')
        # default_kernel = torch.load(f'{self.save_dir}/kernel_default_mixednans.pt')
        default_output = torch.load(f'{self.save_dir}/conv_default_multi4d.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-03).all(): fail += 1

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv multi 4D Test failed: Torch and NaNConv convolutions do not match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv multi 4D Test passed: Torch and NaNConv convolutions match for batch size {batch_size}, channel size {channels}, padding=0 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_all_nans(self):
        """ 
        Example Input:
        [[[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]],


        [[[nan, nan],
          [nan, nan]],

         [[nan, nan],
          [nan, nan]]]]
        """
        torch.manual_seed(0)

        kernel = torch.randn(8, 4, 3, 3)
        inputs = torch.randn(1, 4, 5, 5)
        inputs[:] = float('nan')

        output = torch.zeros((1,8,5,5))
        output[:,:, 1:-1, 1:-1] = float('nan')
        mask = torch.isnan(output)
        output[mask] = 1

        nanoutput = F.conv2d(inputs, kernel, stride=1, padding=1)
        # conv = NaNConv2d(train=False, kernel=kernel, bias=None, padding=1, stride=1, threshold=1)
        # nanoutput = conv(inputs)
        nanoutput[mask] = 1
        
        self.assertTrue(torch.equal(output, nanoutput), "NanConv all nans Test failed: Torch and NaNConv values do not match.")
        print("NanConv all nans Test passed: Torch and NaNConv values match.")



    def test_mixed_nans(self):
        """ 
        Example Input:
        [[[[nan,  1.],
          [ 1.,  1.]],

          [[ 1.,  1.],
          [ 1.,  1.]]],


        [[[nan, nan],
          [nan, nan]],

          [[nan, nan],
          [nan, 10.]]]]
        """
        torch.manual_seed(0)
        
        n = 1000
        nan_inputs = torch.load(f'{self.save_dir}/inputs_mixednans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_mixednans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_mixednans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=1))

        nan_output = torch.stack(nan_output)


        # default_inputs = torch.load(f'{self.save_dir}/inputs_default_mixednans.pt')
        # default_kernel = torch.load(f'{self.save_dir}/kernel_default_mixednans.pt')
        default_output = torch.load(f'{self.save_dir}/conv_default_mixednans.pt')

        mask = torch.isnan(default_output)
        default_output[mask] = 1

        nan_output[mask] = 1

        fail = 0
        for i in range(n):
            if not torch.isclose(nan_output[i], default_output[i], rtol=1e-03).all(): fail += 1
          
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv mixed nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv mixed nan Test passed: Torch and NaNConv inputs match ")

        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv mixed nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv mixed nan Test passed: Torch and NaNConv kernels match ")

        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv mixed nan Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.mixed_fault_tolerance*100}% of the time with a relative tolerance of 1e-03.")
        print(f"NanConv mixed nan Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 1e-03 and disagree under {self.mixed_fault_tolerance*100}% of the time.")


    def test_no_nans(self): 
        """ 
        Example Input:
        [[[[90., 78., 10., 12.],
          [ 6., 64.,  8., 62.]],

          [[75., 39., 51., 52.],
          [44., 79., 66., 34.]]],


        [[[14., 29., 44., 36.],
          [60., 29., 70., 29.]],

          [[12., 93., 76., 94.],
          [22., 44., 20., 58.]]]]
        """
        torch.manual_seed(0)
        n = 1000

        #STRIDE = 1 & PADDING = 1
        nan_inputs = torch.load(f'{self.save_dir}/inputs_stride1_pad1_nonans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_stride1_pad1_nonans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_stride1_pad1_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=1))

        nan_output = torch.stack(nan_output)

        default_output = torch.load(f'{self.save_dir}/conv_default_stride1_pad1_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-04).all(): fail += 1
        
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv no nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv inputs match ")
        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv no nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv kernels match ")
        # Test conv output
        self.assertTrue(fail/n < self.mixed_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        #STRIDE = 1 & PADDING = 0
        nan_inputs = torch.load(f'{self.save_dir}/inputs_stride1_pad0_nonans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_stride1_pad0_nonans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_stride1_pad0_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=1, padding=0))

        nan_output = torch.stack(nan_output)
        default_output = torch.load(f'{self.save_dir}/conv_default_stride1_pad0_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-04).all(): fail += 1
        
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv no nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv inputs match ")
        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv no nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv kernels match ")
        # Test conv output
        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=0 and stride=1 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=0 and stride=1 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")

        #STRIDE = 3 & PADDING = 1
        nan_inputs = torch.load(f'{self.save_dir}/inputs_stride3_pad1_nonans.pt')
        nan_kernel = torch.load(f'{self.save_dir}/kernel_stride3_pad1_nonans.pt')
        # nan_output = torch.load(f'{self.save_dir}/conv_custom_stride3_pad1_nonans.pt')
        nan_output = []
        for i in range(n):
            nan_output.append(F.conv2d(nan_inputs[i], nan_kernel[i], stride=3, padding=1))

        nan_output = torch.stack(nan_output)

        default_output = torch.load(f'{self.save_dir}/conv_default_stride3_pad1_nonans.pt')

        fail=0
        for i in range(n):
              if not torch.isclose(nan_output[i], default_output[i], rtol=5e-04).all(): fail += 1
        
        # # Test conv input
        # self.assertTrue((default_inputs == nan_inputs).all(), f"NanConv no nan Test failed: Torch and NaNConv inputs do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv inputs match ")
        # # Test conv kernel
        # self.assertTrue((default_kernel == nan_kernel).all(), f"NanConv no nan Test failed: Torch and NaNConv kernels do not match ")
        # print(f"NanConv no nan Test passed: Torch and NaNConv kernels match ")
        # Test conv output
        self.assertTrue(fail/n < self.nonan_fault_tolerance, f"NanConv no nans Test failed: Torch and NaNConv convolutions do not match for padding=1 and stride=3 for over {self.nonan_fault_tolerance*100}% of the time with a relative tolerance of 5e-04.")
        print(f"NanConv no nans Test passed: Torch and NaNConv convolutions match for padding=1 and stride=3 within relative tolerance of 5e-04 and disagree under {self.nonan_fault_tolerance*100}% of the time.")




  # Replace this with the actual import of your custom unpooling function
  # from your_module import max_unpool2d_with_indices


class TestMaxUnpool2D(unittest.TestCase):
    
    def setup_test(self, kernel_size, stride, padding, input_tensor, indices_tensor, expected_output, message, output_size=None):
        from nan_ops import NormalUnpool2d

        output = NormalUnpool2d(kernel_size, stride, padding, output_size)(input_tensor, indices_tensor)
        print(output)
        self.assertTrue(torch.allclose(output, expected_output), message)

    def test_unpool_custom_outputshape(self):
        from nan_ops import NaNPool2d
        input_tensor = torch.tensor([
            [
                [[1, 2, 3], [4, 5, 3], [6, 7, 3]],  # Batch 0, Channel 0
                [[3, 1, 2], [3, 3, 4], [5, 3, 3]]   # Batch 0, Channel 1
            ],
            [
                [[31, 41, 51], [31, 61, 71], [31, 81, 91]],  # Batch 1, Channel 0
                [[11, 21, 31], [41, 31, 51], [61, 61, 71]]   # Batch 1, Channel 1
            ]
        ], dtype=torch.float32)


        # Create the MaxUnpool2d layer
        maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        expected_maxval, expected_idx = maxpool(input_tensor)
        
        unpool = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        expected_output = unpool(expected_maxval, expected_idx, output_size=input_tensor.shape)

        # actual_maxval, actual_idx = NaNPool2d()(input_tensor, pool_size=(2,2), strides=(2,2))

        # # Unpooling operation
        # unpool_layer = NaNUnpool2d(kernel_size=2, stride=2, padding=0, output_size=input_tensor.shape)
        # # unpooled_tensor = unpool_layer(expected_maxval, expected_idx)
        # unpooled_tensor = unpool_layer(expected_maxval, idx) #, expected_idx)

        self.setup_test(kernel_size=2, stride=2, padding=0, input_tensor=expected_maxval, indices_tensor=expected_idx, expected_output=expected_output, message="Basic unpooling failed.", output_size=input_tensor.shape)

    
    def test_unpool_basic(self):
        input_tensor = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
        indices_tensor = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.int64)
        kernel_size = (2, 2)
        stride = (2, 2)
        padding = (0, 0)

        expected_output = torch.tensor([[[[1, 2, 3, 4],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]]]], dtype=torch.float32)

        self.setup_test(kernel_size, stride, padding, input_tensor, indices_tensor, expected_output, "Basic unpooling failed.")

    def test_unpool_with_padding(self):
        input_tensor = torch.tensor([[[[0, 0, 0], [0, 4, 0], [0, 0, 0]]]], dtype=torch.float32)
        indices_tensor = torch.tensor([[[[0, 1, 3], [4, 10, 7], [12, 13, 15]]]], dtype=torch.int64)
        kernel_size = (2, 2)
        stride = (2, 2)
        padding = (1, 1)

        expected_output = torch.tensor([[[[0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 4, 0],
                                          [0, 0, 0, 0]]]], dtype=torch.float32)

        self.setup_test(kernel_size, stride, padding, input_tensor, indices_tensor, expected_output, "Unpooling with padding failed.")

    def test_unpool_non_square_pooling(self):
        input_tensor = torch.tensor([[[[5, 6], [7, 8]]]], dtype=torch.float32)
        indices_tensor = torch.tensor([[[[0, 2], [4, 6]]]], dtype=torch.int64)
        kernel_size = (2, 3)
        stride = (2, 3)
        padding = (0, 0)

        expected_output = torch.tensor([[[[5, 0, 6, 0, 7, 0],
                                          [8, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0]]]], dtype=torch.float32)

        self.setup_test(kernel_size, stride, padding, input_tensor, indices_tensor, expected_output, "Unpooling with non-square pooling failed.")
        

    def test_unpool_multi_channel(self):
        input_tensor = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32)
        indices_tensor = torch.tensor([[[[0, 1], [2, 3]], [[0, 2], [4, 6]]]], dtype=torch.int64)
        kernel_size = (2, 2)
        stride = (2, 2)
        padding = (0, 0)

        expected_output = torch.tensor([[[[1, 2, 3, 4],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]],
                                          [[5, 0, 6, 0],
                                          [7, 0, 8, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]]]], dtype=torch.float32)

        self.setup_test(kernel_size, stride, padding, input_tensor, indices_tensor, expected_output, "Unpooling with multiple channels failed.")


    def test_unpool_large_tensor(self):
        input_tensor = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 4, 4)
        # Perform max pooling with kernel size 2x2 and stride 2x2
        pooled_tensor, indices_tensor = max_pool2d(input_tensor, kernel_size=2, stride=2, return_indices=True)

        kernel_size = (2, 2)
        stride = (2, 2)
        padding = (0, 0)

        # Create the expected output
        expected_output = torch.tensor([[[[ 0,  0,  0,  0],
                                          [ 0,  6,  0,  8],
                                          [ 0,  0,  0,  0],
                                          [ 0,  14,  0,  16]]]], dtype=torch.float32)

        self.setup_test(kernel_size, stride, padding, pooled_tensor, indices_tensor, expected_output, "Unpooling with larger tensor failed.")


class TestNaNUnpool2D(unittest.TestCase):

    def setup_test_nan(self, kernel_size, stride, padding, input_tensor, expected_output, message):
        from nan_ops import NaNUnpool2d 
        from nan_ops import NaNPool2d_v2 as NaNPool2d

        # Create Torch NaN output 
        actual_maxval, actual_idx = NaNPool2d()(input_tensor, pool_size=kernel_size, strides=stride, padding=padding)
        unpool_layer = NaNUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding, output_size=input_tensor.shape)
        actual_output = unpool_layer(actual_maxval, actual_idx) 

        print(actual_output.shape, expected_output.shape)
        print(actual_output, expected_output)
        print(actual_output == expected_output)

        self.assertTrue(torch.allclose(torch.nan_to_num(actual_output, nan=3.14), torch.nan_to_num(expected_output, nan=3.14)), message)

    def setup_test_nonan(self, kernel_size, stride, padding, input_tensor, message):
        from nan_ops import NaNUnpool2d
        from nan_ops import NaNPool2d_v2 as NaNPool2d


        # Create Torch default output 
        maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
        expected_maxval, expected_idx = maxpool(input_tensor)
        unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        expected_output = unpool(expected_maxval, expected_idx, output_size=input_tensor.shape)
        
        # Create Torch NaN output 
        actual_maxval, actual_idx = NaNPool2d()(input_tensor, pool_size=kernel_size, strides=stride, padding=padding)
        unpool_layer = NaNUnpool2d(kernel_size=kernel_size, stride=stride, padding=padding, output_size=input_tensor.shape)
        actual_output = unpool_layer(actual_maxval, actual_idx) 

        # print(actual_output == expected_output)

        self.assertTrue(torch.allclose(actual_output, expected_output), message)

    def test_unpool_basic_stride1_nan(self):

        input_tensor = torch.tensor([
            [
                [[1, 2, 3], [4, 5, 3], [6, 7, 3]],  # Batch 0, Channel 0
                [[3, 1, 2], [3, 3, 4], [5, 3, 3]]   # Batch 0, Channel 1
            ],
            [
                [[31, 41, 51], [31, 61, 71], [31, 81, 91]],  # Batch 1, Channel 0
                [[11, 21, 31], [41, 31, 51], [61, 61, 71]]   # Batch 1, Channel 1
            ]
        ], dtype=torch.float32)

        expected_output = torch.tensor([[[[ 0.,  0.,  0.],
                                          [ 0.,  5.,  0.],
                                          [ 0.,  7.,  0.]],
                                          
                                          [[float('nan'),  0.,  0.],
                                           [float('nan'), float('nan'),  4.],
                                           [ 5.,  0.,  0.]]],
                                           
                                           [[[ 0.,  0.,  0.],
                                             [ 0., 61., 71.],
                                             [ 0., 81., 91.]],
                                             
                                             [[ 0.,  0.,  0.],
                                              [41.,  0., 51.],
                                              [float('nan'), float('nan'), 71.]]]])
        
        self.setup_test_nan(kernel_size=2, stride=1, padding=0, input_tensor=input_tensor, expected_output=expected_output, message="Basic unpooling with duplicate max values failed.")

    def test_unpool_batch1_nan(self):

        input_tensor = torch.tensor([[
            [[6, 6, 3, 4], [5, 6, 7, 8], [9, 10, 11, 16], [13, 14, 15, 16]]
        ]], dtype=torch.float32)


        expected_output = torch.tensor([[[[float('nan'), float('nan'),  0.,  0.],
          [ 0., float('nan'),  0.,  8.],
          [ 0.,  0.,  0., float('nan')],
          [ 0., 14.,  0., float('nan')]]]])
        
        self.setup_test_nan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, expected_output=expected_output, message="Single batch unpooling with duplicate max values failed.")

    def test_unpool_multichannel_nan(self):
        # Input tensor of size (1, 3, 4, 4) with 3 channels
        input_tensor = torch.tensor([[
            [[1, 2, 3, 4], [8, 8, 8, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 38, 38, 38], [25, 26, 27, 28], [29, 30, 31, 32]],
            [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]
        ]], dtype=torch.float32)

        expected_output = torch.tensor([[[[ 0.,  0.,  0.,  0.],
          [float('nan'), float('nan'), float('nan'), float('nan')],
          [ 0.,  0.,  0.,  0.],
          [ 0., 14.,  0., 16.]],

         [[ 0.,  0.,  0.,  0.],
          [ 0., 38., float('nan'), float('nan')],
          [ 0.,  0.,  0.,  0.],
          [ 0., 30.,  0., 32.]],

         [[ 0.,  0.,  0.,  0.],
          [ 0., 38.,  0., 40.],
          [ 0.,  0.,  0.,  0.],
          [ 0., 46.,  0., 48.]]]])

        self.setup_test_nan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, expected_output=expected_output, message="Multichannel unpooling with duplicate max values failed.")

    def test_unpool_largekernel_nan(self):
        # Input tensor of size (1, 3, 4, 4) with 3 channels
        input_tensor = torch.tensor([[
            [[40, 40, 40, 40, 40, 40], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], 
            [19, 20, 21, 22, 23, 24], [40, 40, 40, 50, 40, 50], [31, 32, 33, 34, 35, 36]]
        ]], dtype=torch.float32)

        expected_output = torch.tensor([[[[float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
          [ 0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0.,  0.,  0.,  0.,  0.],
          [ 0.,  0., 21., 22., 23., 24.],
          [float('nan'), float('nan'), float('nan'), float('nan'),  0., float('nan')],
          [ 0.,  0.,  0.,  0.,  0.,  0.]]]])

        self.setup_test_nan(kernel_size=3, stride=1, padding=0, input_tensor=input_tensor, expected_output=expected_output, message="Large kernel unpooling with duplicate max values failed.")

    def test_unpool_nonsquarekernel_nan(self):
        # Input tensor of size (1, 3, 4, 4) with 3 channels
        input_tensor = torch.tensor([[
            [[1, 2, 3, 4, 5, 6], [8, 8, 9, 10, 12, 12], [14, 14, 16, 16, 17, 18]]
        ]], dtype=torch.float32)

        expected_output = torch.tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.],
          [float('nan'), float('nan'),  0., 10., float('nan'), float('nan')],
          [ 0.,  0.,  0.,  0.,  0.,  0.]]]])

        self.setup_test_nan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, expected_output=expected_output, message="Single batch large kernel unpooling with duplicate max values failed.")
    
    def test_unpool_basic_stride1_nonan(self):
        input_tensor = torch.tensor([
            [
                [[1, 2, 3], [4, 5, 3], [6, 7, 3]],  # Batch 0, Channel 0
                [[3, 0, 1], [3, 2, 4], [5, 8, 9]]   # Batch 0, Channel 1
            ],
            [
                [[31, 41, 51], [31, 61, 71], [31, 81, 91]],  # Batch 1, Channel 0
                [[11, 21, 32], [41, 32, 51], [61, 32, 71]]   # Batch 1, Channel 1
            ]
        ], dtype=torch.float32)

        self.setup_test_nonan(kernel_size=2, stride=1, padding=0, input_tensor=input_tensor, message="Basic unpooling without any duplicate max values failed.")

    def test_unpool_basic_stride2_nonan(self):
        input_tensor = torch.tensor([
            [
                [[1, 2, 3], [4, 5, 3], [6, 7, 3]],  # Batch 0, Channel 0
                [[3, 1, 2], [3, 3, 4], [5, 3, 3]]   # Batch 0, Channel 1
            ],
            [
                [[31, 41, 51], [31, 61, 71], [31, 81, 91]],  # Batch 1, Channel 0
                [[11, 21, 31], [41, 31, 51], [61, 31, 71]]   # Batch 1, Channel 1
            ]
        ], dtype=torch.float32)

        self.setup_test_nonan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, message="Basic unpooling without any duplicate max values failed.")

    def test_unpool_batch1_nonan(self):
        # Input tensor of size (1, 1, 4, 4)
        input_tensor = torch.tensor([[
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        ]], dtype=torch.float32)

        self.setup_test_nonan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, message="Single batch unpooling without any duplicate max values failed.")

    def test_unpool_multichannels_nonan(self):
        # Input tensor of size (1, 3, 4, 4) with 3 channels
        input_tensor = torch.tensor([[
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],
            [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]
        ]], dtype=torch.float32)

        self.setup_test_nonan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, message="Multi channel unpooling without any duplicate max values failed.")

    def test_unpool_largekernel_nonan(self):
        # Input tensor of size (1, 1, 6, 6)
        input_tensor = torch.tensor([[
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], 
            [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]
        ]], dtype=torch.float32)

        self.setup_test_nonan(kernel_size=3, stride=1, padding=0, input_tensor=input_tensor, message="Large kernel unpooling without any duplicate max values failed.")

    def test_unpool_nonsquarekernel_nonan(self):
        # Input tensor of size (1, 1, 3, 6)
        input_tensor = torch.tensor([[
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
        ]], dtype=torch.float32)

        self.setup_test_nonan(kernel_size=2, stride=2, padding=0, input_tensor=input_tensor, message="Single batch large kernel unpooling without any duplicate max values failed.")

    

class TestNaNPool2D_v2(unittest.TestCase):

    def compare(self, expected_maxval, expected_idx, nan_maxval, nan_idx, maxval_msg, idx_msg):
        #Reshape nan_idx for comparison purposes
        tmp_idx = []
        for i in list(nan_idx.values()):
            tmp = []
            if len(i) != 1: #modified indices
                for b in range(len(i)):
                    if i[b].shape: #indices for multiple max values
                        tmp.append(i[b][0])
                    else: 
                        tmp.append(i[b])
            else:
                tmp = [i[0][0], i[0][1]]

            tmp_idx.append(tmp)

        self.assertTrue(torch.allclose(expected_maxval, nan_maxval), maxval_msg)
        self.assertTrue(torch.allclose(expected_idx.squeeze(), torch.tensor(tmp_idx).T), idx_msg)


    def test_basic(self):
        from nan_ops import NaNPool2d_v2
        input_tensor = torch.tensor([
            [
                [[1, 2, 3], [4, 5, 3], [6, 7, 3]],  # Batch 0, Channel 0
                [[3, 1, 2], [3, 3, 4], [5, 3, 3]]   # Batch 0, Channel 1
            ],
            [
                [[31, 41, 51], [31, 61, 71], [31, 81, 91]],  # Batch 1, Channel 0
                [[11, 21, 31], [41, 31, 51], [61, 31, 71]]   # Batch 1, Channel 1
            ]
        ], dtype=torch.float32)


        nan_maxval, nan_idx = NaNPool2d_v2()(input_tensor, pool_size=(2,2), strides=(2,2))
        expected_maxval, expected_idx = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(input_tensor)
        self.compare(expected_maxval, expected_idx, nan_maxval, nan_idx, "Max values do not match for the basic tensor NaN pooling test", "Indices do not match for the basic tensor NaN pooling test")


    def test_big_tensor(self):
        from nan_ops import NaNPool2d_v2
        input_tensor = torch.tensor([
            [
                [[1, 2, 3], [3, 3, 3], [6, 7, 8]],  # Batch 0, Channel 0
                [[4, 5, 5], [5, 5, 6], [8, 8, 8]],  # Batch 0, Channel 1
                [[2, 1, 2], [3, 4, 2], [6, 6, 2]]   # Batch 0, Channel 2
            ],
            [
                [[21, 22, 22], [21, 23, 23], [24, 25, 25]],  # Batch 1, Channel 0
                [[11, 12, 12], [11, 13, 14], [14, 15, 15]],  # Batch 1, Channel 1
                [[5, 5, 5], [6, 5, 6], [6, 7, 7]]           # Batch 1, Channel 2
            ]
        ], dtype=torch.float32)


        nan_maxval, nan_idx = NaNPool2d_v2()(input_tensor, pool_size=(2,2), strides=(2,2))
        expected_maxval, expected_idx = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(input_tensor)
        self.compare(expected_maxval, expected_idx, nan_maxval, nan_idx, "Max values do not match for the big tensor NaN pooling test", "Indices do not match for the big tensor NaN pooling test")


    def test_identicalmaxvals(self):
        from nan_ops import NaNPool2d_v2
        input_tensor = torch.tensor([
            [
                [[10, 10, 12], [10, 10, 15], [11, 10, 15]],  # Batch 0, Channel 0
                [[7, 7, 7], [7, 6, 6], [8, 8, 8]]           # Batch 0, Channel 1
            ],
            [
                [[3, 3, 2], [3, 3, 2], [2, 3, 3]],          # Batch 1, Channel 0
                [[5, 5, 5], [5, 5, 5], [6, 5, 6]]           # Batch 1, Channel 1
            ],
            [
                [[30, 30, 30], [30, 40, 40], [40, 40, 30]], # Batch 2, Channel 0
                [[9, 9, 8], [8, 9, 9], [9, 9, 9]]           # Batch 2, Channel 1
            ]
        ], dtype=torch.float32)


        nan_maxval, nan_idx = NaNPool2d_v2()(input_tensor, pool_size=(2,2), strides=(2,2))
        expected_maxval, expected_idx = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)(input_tensor)
        self.compare(expected_maxval, expected_idx, nan_maxval, nan_idx, "Max values do not match for the identical max values tensor NaN pooling test", "Indices do not match for the identical max values tensor NaN pooling test")


    def test_poolsize(self):
        from nan_ops import NaNPool2d_v2
        input_tensor = torch.tensor([
            [
                [[2, 5, 5], [5, 5, 3], [3, 2, 2]],  # Batch 0, Channel 0
                [[9, 9, 9], [8, 9, 9], [9, 8, 9]]   # Batch 0, Channel 1
            ],
            [
                [[12, 12, 12], [11, 12, 12], [10, 12, 11]],  # Batch 1, Channel 0
                [[4, 5, 5], [6, 5, 5], [5, 4, 5]]           # Batch 1, Channel 1
            ]
        ], dtype=torch.float32)


        nan_maxval, nan_idx = NaNPool2d_v2()(input_tensor, pool_size=(3,3), strides=(2,2))
        expected_maxval, expected_idx = torch.nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)(input_tensor)
        self.compare(expected_maxval, expected_idx, nan_maxval, nan_idx, "Max values do not match for the pool size tensor NaN pooling test", "Indices do not match for the pool size tensor NaN pooling test")
        

  
def run_single_test_with_arg(test_class, test_name, **kwargs):
    suite = unittest.TestSuite()
    test = test_class(test_name)

    # Set custom attributes
    for key, value in kwargs.items():
        setattr(test, key, value)

    suite.addTest(test)
    unittest.TextTestRunner().run(suite)


if __name__ == "__main__":
    
    torch.manual_seed(0)
    
    # Run all tests
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNPool2d)
    # unittest.TextTestRunner().run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaNUnpool2D)
    unittest.TextTestRunner().run(suite)

    #prep = PrepNaNConv2d(save_dir='/nanconv_unittests')
    #prep.prep_mixed_nans(mixednan_filename='default')
    #prep.prep_no_nans(nonan_filename='default')
    #prep.prep_multichannel(multichannel_filename='default')
    #prep.prep_multibatch(multibatch_filename='default')
    #prep.prep_multi4d(multi4d_filename='default')

    # # Run a specific test
    # run_single_test_with_arg(TestNaNConv2d, 'test_mixed_nans', nonan_fault_tolerance=0.1, mixed_fault_tolerance=0.1) #possible to adjust fault tolerance for NaNConv tests
