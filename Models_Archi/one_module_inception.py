import torch 
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchview import draw_graph
from torchviz import make_dot


def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


def pass_through(X):
	return X


class Inception(nn.Module):
	def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
		"""
		super(Inception, self).__init__()
		self.return_indices=return_indices
		if in_channels > 2:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels, 
								out_channels=bottleneck_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = in_channels  #1

		self.conv_from_bottleneck_1 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=(kernel_sizes[0] -1 )//2,   #p=k-1/2
										bias=False
										)
		self.conv_from_bottleneck_2 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=(kernel_sizes[1]-1)//2, 
										bias=False
										)
		self.conv_from_bottleneck_3 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=(kernel_sizes[2]-1)//2, 
										bias=False
										)
		self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
		self.conv_from_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=n_filters, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
		self.activation = activation

	def forward(self, X):
		# step 1
		Z_bottleneck = self.bottleneck(X)
		if self.return_indices:
			Z_maxpool, indices = self.max_pool(X)
		else:
			Z_maxpool = self.max_pool(X)
		# step 2
		Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
		Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
		Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
		Z4 = self.conv_from_maxpool(Z_maxpool)
		# step 3 
		Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
		Z = self.activation(self.batch_norm(Z))
		if self.return_indices:
			return Z, indices
		else:
			return Z

class AttentionGate1D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(1),
            nn.Sigmoid() 
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  
        psi = self.psi(psi)    
        return x * psi  
	

class InceptionBlock(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False, use_attention=False):
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
		self.use_attention = use_attention  
		self.return_indices = return_indices
		self.activation = activation
		self.inception_1 = Inception(
							in_channels=in_channels,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)


		if self.use_residual:
			self.residual = nn.Sequential(
               					nn.Conv1d(
									in_channels=in_channels, 
									out_channels=4*n_filters, 
									kernel_size=1,
									stride=1,
									padding=0
									),
               					nn.BatchNorm1d(
									num_features=4*n_filters
									)
          						)
			
			#zjout de l'Attention Gate
			if self.use_attention: 
				self.attention = AttentionGate1D(
                    F_g = 4*n_filters,  # taille des features profondes (sortie inception block)
                    F_l = 4*n_filters,  # taille des features résiduelles
                    F_int = bottleneck_channels  # dimension intermédiaire
                )
		

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
		else:
			Z = self.inception_1(X)

		#connexion résiduelle avec attention
		if self.use_residual:
			residual = self.residual(X)
			if self.use_attention:
				residual = self.attention(g=Z, x=residual)  
			Z = Z + residual
			Z = self.activation(Z)

		if self.return_indices:
			return Z, [i1]
		
		return Z




class InceptionTimeModel_1(nn.Module):
    def __init__(self, 
        in_channels_1,  # Input channels for time series 1 ( rainfall)
        in_channels_2,  # Input channels for time series 2 ( temperature)
        num_classes,  n_blocks=1, n_filters=32,  kernel_sizes=[9, 19, 39],  bottleneck_channels=32, use_residual=True, nb_branches=2,
        merge_mode='concat', use_attention=True):

        super(InceptionTimeModel_1, self).__init__()
        self.merge_mode = merge_mode
        
        #Branch 1 (for time series 1)
        self.branch_1 = nn.ModuleList()
        for i in range(n_blocks):
            block_in_channels = in_channels_1 if i == 0 else 4 * n_filters
            self.branch_1.append(
                InceptionBlock(
                    in_channels=block_in_channels,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual,
					use_attention=use_attention
				) 
			)
			
        #Branch 2 (for time series 2)
        if nb_branches == 2 :
            self.branch_2 = nn.ModuleList()
            for i in range(n_blocks):
                block_in_channels = in_channels_2 if i == 0 else 4 * n_filters
                self.branch_2.append(
                    InceptionBlock(
                        in_channels=block_in_channels,
                        n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							use_residual=use_residual,
							use_attention=use_attention
						)
					)

        #Merge features
        if self.merge_mode == 'concat':
            merged_features = 4 * n_filters * 2  #2 for number of modalities
        # elif self.merge_mode == 'add':
        #     merged_features = 4 * n_filters  #reste same shape if adding
        else:
            raise ValueError("merge_mode must be 'concat' or 'add'")
		
        # Global average pooling(shared)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)        
        #add a dense layer after merging
        #self.merge_fc = nn.Linear(merged_features, 4 * n_filters)
        self.merge_fc = nn.Sequential(
			nn.Linear(merged_features, 4 * n_filters),
			nn.ReLU() )
        # Fully connected layer for classification
        self.fc = nn.Linear(4 * n_filters, num_classes)

    def forward(self, x1, x2):
        #Process branch 1 (time series 1)
        for block in self.branch_1:
            x1 = block(x1)
        x1 = self.global_avg_pool(x1).squeeze(-1)  #[batch, 4*n_filters]
        
        #Process branch 2 (time series 2)
        if x2 is not None :
            for block in self.branch_2:
              x2 = block(x2)
            x2 = self.global_avg_pool(x2).squeeze(-1)  #[batch, 4*n_filters]

			#Merge branches
            if self.merge_mode == 'concat':
               x = torch.cat([x1, x2], dim=-1)  #[batch, 4*n_filters*2]
            elif self.merge_mode == 'add':
               x = x1 + x2  # Shape: [batch, 4*n_filters]

			#Pass through a dense layer in case of concat
            x = self.merge_fc(x)
        
		    # Final classification
            x = self.fc(x)
            return x
		
	    # Final classification
        x = self.fc(x1)
        return x
		
model = InceptionTimeModel_1(
    in_channels_1=1,  #time series (1 channel)
    in_channels_2=1,  #second time series (1 channel)
    num_classes=2,  
    merge_mode='concat' , 
	kernel_sizes=[9,19,39],
	use_attention=False
)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params}")
print(f"Trainable params: {trainable_params}")


