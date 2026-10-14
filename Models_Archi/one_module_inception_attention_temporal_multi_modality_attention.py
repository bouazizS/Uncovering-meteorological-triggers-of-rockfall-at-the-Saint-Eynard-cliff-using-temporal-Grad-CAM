import torch 
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchview import draw_graph
from torchviz import make_dot

''' multi inception block then fusion'''

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
		if in_channels > 32:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels, 
								out_channels=bottleneck_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = in_channels

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

class ModalitySelfAttention(nn.Module):

    def __init__(self, feature_dim, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True    # expects [B, seq, F]
        )
        # Optional: project back to feature_dim after attention
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, branch_vectors):
        # Stack modality vectors: [B, M, F]
        H = torch.stack(branch_vectors, dim=1)

        # Self-attention: each modality attends to all others
        # attn_output: [B, M, F],  attn_weights: [B, M, M]
        attn_output, attn_weights = self.attn(H, H, H)

        # Residual + norm (standard transformer practice)
        attn_output = self.norm(attn_output + H)

        # Pool across modality dimension → [B, F]
        # mean-pool is simple; you could also take the CLS-style first token
        out = attn_output.mean(dim=1)

        return out, attn_weights   # attn_weights[b, i, j] = how much modality i attends to j
	
	
class InceptionBlock(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False):
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
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
			

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
		else:
			Z = self.inception_1(X)

		if self.use_residual:
			residual = self.residual(X)
			Z = Z + residual
			Z = self.activation(Z)

		if self.return_indices:
			return Z, [i1]
		
		return Z


kernel_sizes_list = [
    [19, 51, 99],  # Rain
    [9, 19, 39],   # Temp
    [19, 51, 99],  # Charge
    [9, 19, 39]    # Freeze
]

class InceptionTimeModelAttention(nn.Module):
    def __init__(self, 
        in_channels_list,  # list of input channels per modality 
        num_classes,  n_blocks=1, n_filters=32,  kernel_sizes= kernel_sizes_list,  bottleneck_channels=32, use_residual=True,
        merge_mode='concat', use_modality_attention=True):

        super(InceptionTimeModelAttention, self).__init__()
        self.merge_mode = merge_mode
        self.num_modalities = len(in_channels_list)
        self.use_modality_attention = use_modality_attention
		#create one branch for each input modality
        self.branches = nn.ModuleList()
        for mod_idx, in_channels in enumerate(in_channels_list):  # ← enumerate
            branch = nn.ModuleList()
            for i in range(n_blocks):
                block_in = in_channels if i == 0 else 4 * n_filters
                block = InceptionBlock(
                    in_channels=block_in,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes[mod_idx], 
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual
                )
                branch.append(block)
            self.branches.append(branch)

        #Merge features
        if self.merge_mode == 'concat':
            merged_features = 4 * n_filters * self.num_modalities  #2 for number of modalities
        # elif self.merge_mode == 'add':
        #     merged_features = 4 * n_filters  #reste same shape if adding
        else:
            raise ValueError("merge_mode must be 'concat' or 'add'")
		
        # Global average pooling(shared)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  

        # If using modality attention, the merger input size is 4*n_filters (attended vector)
        merge_in_features = 4 * n_filters if use_modality_attention else merged_features

        #add a dense layer after merging
        self.merge_fc = nn.Sequential(
			nn.Linear(merge_in_features, 4 * n_filters),
			nn.ReLU() )
		
        if self.use_modality_attention:
            self.modality_attention = ModalitySelfAttention(feature_dim=4 * n_filters, num_heads=1)

        self.fc = nn.Linear(4 * n_filters, num_classes)

    def forward(self, *inputs):  #*inputs : receiving a variable number of inputs
        assert len(inputs) == self.num_modalities, f"Expected {self.num_modalities} inputs but got {len(inputs)}"
        branch_outputs = []
        for i, (x, branch) in enumerate(zip(inputs, self.branches)):
            for block in branch:
                x = block(x)
            x = self.global_avg_pool(x).squeeze(-1)
            branch_outputs.append(x)

        attn_weights = None
        if self.use_modality_attention:
            x, attn_weights = self.modality_attention(branch_outputs)
		
        else:
            # default merge: concat along feature dim
            if self.merge_mode == 'concat':
                x = torch.cat(branch_outputs, dim=-1)  #[batch, 4*n_filters*num_modalities]
            else:
                # implement other merge modes if needed
                raise ValueError("unsupported merge_mode without attention")

			#Pass through a dense layer
            x = self.merge_fc(x)

		# Final classification
        x = self.fc(x)

        return x , attn_weights
		



class ModalityAttention(nn.Module):

    def __init__(self, num_modalities, feature_dim):
        super().__init__()
        # Small MLP: per-modality feature vector → scalar score
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)   # one score per modality
        )

    def forward(self, branch_vectors):
        # branch_vectors: list of [B, F]  →  stack to [B, M, F]
        H = torch.stack(branch_vectors, dim=1)          # [B, M, F]
        scores = self.gate(H).squeeze(-1)               # [B, M]
        alpha  = torch.softmax(scores, dim=-1)          # [B, M]  sums to 1
        # Weighted sum across modalities
        out = (alpha.unsqueeze(-1) * H).sum(dim=1)      # [B, F]
        return out, alpha
    


class InceptionTimeModelAttentionMLP(nn.Module):
    def __init__(self,
                 in_channels_list,
                 num_classes,
                 n_blocks=1,
                 n_filters=32,
                 merge_mode='concat',
                 kernel_sizes=kernel_sizes_list,
                 bottleneck_channels=32,
                 use_residual=True,
                 use_modality_attention=True):  

        super().__init__()
        self.num_modalities = len(in_channels_list)
        self.use_modality_attention = use_modality_attention
        self.merge_mode = merge_mode

        # One inception branch per modality (unchanged)
        self.branches = nn.ModuleList()
        for mod_idx, in_channels in enumerate(in_channels_list):  # ← enumerate
            branch = nn.ModuleList()
            for i in range(n_blocks):
                block_in = in_channels if i == 0 else 4 * n_filters
                block = InceptionBlock(
                    in_channels=block_in,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes[mod_idx], 
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual
                )
                branch.append(block)
            self.branches.append(branch)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        feature_dim = 4 * n_filters

        # Point 4: modality attention replaces concat + merge_fc
        if use_modality_attention:
            self.modality_attention = ModalityAttention(
                num_modalities=self.num_modalities,
                feature_dim=feature_dim
            )
            self.fc = nn.Linear(feature_dim, num_classes)
        else:
            # Fallback: keep old concat behaviour for comparison
            self.merge_fc = nn.Sequential(
                nn.Linear(feature_dim * self.num_modalities, feature_dim),
                nn.ReLU()
            )
            self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, *inputs):
        assert len(inputs) == self.num_modalities

        # Run each branch
        branch_outputs = []
        for x, branch in zip(inputs, self.branches):
            for block in branch:
                x = block(x)
            x = self.global_avg_pool(x).squeeze(-1)   # [B, F]
            branch_outputs.append(x)

        if self.use_modality_attention:
            # Returns weighted representation + scores [B, M]
            x, self.attention_scores = self.modality_attention(branch_outputs)
        elif self.merge_mode == 'concat':
            x = torch.cat(branch_outputs, dim=-1)
            x = self.merge_fc(x)

        return self.fc(x) , self.attention_scores



