import pandas as pd, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CombinationOHETransformer(BaseEstimator, TransformerMixin):
	"""Transforms two adjacent categorical columns containing same categories into one set of one hot encoded columns containing either binary or aggregate values"""
	
	
	def __init__(self,columns, rem_cols=None, agg_cols=None, comb=None, comb_col=None, proportion=1):
		"""
		Parameters:
		columns : list 
			a list containing two columns whose string values will be transformed into column names
		rem_cols: str or list
			a string or a list of strings representing values from the original columns that should not be transformed into new columns
		agg_cols : list 
			the aggregate columns corresponding to columns. The values in the original columns will be mapped to the transformed columns
		comb : list
			a list of strings that can be values in the column parameter that should be combined into a single column rather than separate values.
		comb_col: str
			the name of the combined column
		proportion: str (default=1)
			column name used to normalize agg_cols
		"""
		
		self.org_columns = columns
		self.columns = columns
		self.rem_cols = rem_cols
		self.agg_cols = agg_cols
		self.proportion = proportion
		self.comb = comb
		self.comb_col = comb_col

	def fit(self, df, y=None):
		
		"""
		fits one hot encodings columns from the columns parameter passed during instantation
		
		Parameters:
		df : the dataframe containing the desired columns to encode
		
		Returns: None
		
		"""
		
		col1 = self.org_columns[0]
		col2 = self.org_columns[1]
		self.columns = list(set(df[col1]).union(set(df[col2])))
		if self.rem_cols is not None:
			self.parse_remove_cols()
		self.X = pd.DataFrame(columns=list(self.columns))
		return None
	
	def transform(self,df,y=None):
		
		"""
		Transforms desired columns in passed in dataframe to onehot encodings in either binary or aggregated forms
		
		Parameters:
		df: the dataframe containing the desired columns created from fit
		"""
		
		X = self.X.copy()
		t = pd.get_dummies(df[self.org_columns],prefix_sep='_')
		if self.agg_cols != None:
			X = self.make_agg(t,X,df)
		else:
			X = self.make_binary(t,X)
		X.fillna(0,inplace=True,axis=1)
		return X
	
	def fit_transform(self, df,y=None):
		
		"""
		Fits and transforms the original dataframe containing the desired columns in either binary or aggregated forms. This shouldn't be called directly.
		
		Parameters:
		df: the dataframe containing the desired columns whose values will be transformed into separate columns
		"""
		
		col1 = self.org_columns[0]
		col2 = self.org_columns[1]
		self.columns = list(set(df[col1]).union(set(df[col2])))
		if self.rem_cols is not None:
			self.parse_remove_cols()
		self.X = pd.DataFrame(columns=list(self.columns))
		X = self.X.copy()
		t = pd.get_dummies(df[self.org_columns],prefix_sep='_')
		if self.agg_cols != None:
			X = self.make_agg(t,X,df)
		else:
			X = self.make_binary(t,X)
		X.fillna(0,inplace=True,axis=1)
		return X
	
	def make_binary(self,new_X,X):
		""" transforms each newly encoded column with a binary value  (1 or 0) to the column set. This is called during the fit.
		
		Parameters:
		new_X : dataframe
			shell dataframe with fitted columns
		X : dataframe
			contains the original columns to be transformed
		
		Returns: new dataframe with fitted columns. Each row corresponds to the original X and has 0 or 1 accordingly.
		"""
		for col in self.X.columns:
			X[col] = new_X.apply(lambda x: 1 if np.sum([x[i] for i in new_X.columns if col in i]) > 0 else 0, axis=1)
		if self.comb != None:
			X[self.comb_col] = X.apply(lambda x: 1 if np.sum(x[self.comb]) > 0 else 0, axis=1)
			X.drop(columns=self.comb, axis=1, inplace=True)
		return X
	
	def make_agg(self,new_X,X,df):
		""" takes data from an adjacent aggregate column and inserts as values \\
		into corresponding ohe column
		
		Parameters:
		new_X: dataframe
			shell dataframe with fitted columns
		X: dataframe
			contains the original columns and data to be transformed	

		returns: dataframe with fitted columns. Each row corresponds to the original X and has 
		the corresponding values specified in agg_col argument at initialization.
		"""
		if type(self.proportion) == str:
			proportion = df[self.proportion]
		else:
			proportion = 1
		
		df1 = (df.pivot(columns=[self.org_columns[0]],values=self.agg_cols[0])
				.fillna(0))
		df2  = (df.pivot(columns=[self.org_columns[1]],values=self.agg_cols[1])
				.fillna(0))
		for col in df2.columns:
			if col in df1.columns:
				df1[col] = (df1[col] + df2[col])/proportion
			else:
				df1[col] = df2[col]/proportion
		return df1

	
	def parse_remove_cols(self):
		"""Remove any columns after one-hot encoding specified at instantiation. This is called based on rem_cols argument and should not be called directly."""
		if type(self.rem_cols) == list:
			self.columns = list(set(self.columns).difference(set(self.rem_cols)))
		elif type(self.rem_cols) == str:
			self.columns.remove(self.rem_cols)
		return None
	
	def set_output(self,transform):
		pass
	
	
class Cat2Val(BaseEstimator,TransformerMixin):
	"""transforms a single categorical column's values into columns \\
	and aggregates values from another column similar to pandas pivot"""
	
	
	def __init__(self, c_col, v_col):
		"""Parameters:
		c_col : str
			represents the name of a categorical column whose values will be transformed into columns
		v_col: str
			represents the name of a int/float column whose values will be added to the transformed columns
		"""		
		
		self.c_col = c_col
		self.v_col = v_col
		self.deep = True
		
	def fit(self, df, y=None):
		
		"""Parameters:
		df: dataframe
			original dataframe containing the columns specified at instantiation to be transformed
		y: None
			not used
		Returns: fitted object
		"""
		
		
		c_col = self.c_col
		self.columns = list(df[c_col].unique())
		return self
	
	def fit_transform(self, df, y=None):
		
		"""Parameters:
		df: dataframe
			original dataframe containing the columns specified at instantiation to be transformed
		y: None
			not used
		Returns: dataframe transformed from original dataframe
		"""

		
		X = df.copy()
		c_col = self.c_col
		v_col = self.v_col
		X = pd.pivot(data=X[[c_col,v_col]],
			  columns=c_col,
			  values=v_col).fillna(0)
		if np.nan in X.columns:
			X.drop(columns=[np.nan], axis=1, inplace=True)
		X.fillna(0, inplace=True)
		X = X.astype('int')
		self.columns = list(X.columns)
		return X
		
		
	def transform(self, df, y=None):
		
		"""Parameters:
		df: dataframe
			original dataframe containing the columns specified at instantiation to be transformed
		y: None
			not used
		Returns: dataframe transformed from original dataframe
		"""
		
		
		
		X = pd.DataFrame(index= df.index, columns = self.columns)
		c_col = self.c_col
		v_col = self.v_col
		for col in self.columns:
			if col in list(df[c_col].unique()):
				X[col] = df.apply(lambda x: x[v_col] if x[c_col]==col else 0, axis=1)
			else:
				X[col] = 0
		if np.nan in X.columns:
			X.drop(columns=[np.nan], axis=1, inplace=True)
		X.fillna(0, inplace=True)
		return X
	
	def set_output(self,transform):
		pass
	

class Cat2Dummies(BaseEstimator,TransformerMixin):
	"""Class similiar in function to pandas get_dummies but allows users to specify \\\
	which categories to keep; automatically creates an 'Other' bin for non-specified columns"""
	
	def __init__(self, c_col, categories):
		self.c_col = c_col
		self.categories = categories
		
		"""Parameters:
		c_col: str
			name of a column to be transformed into a dummy column
		categories: list
			list of categories to keep once dummy columns created. All other columns will be added to "Other".
		"""
		
		
	def fit(self, df, y=None):
		# X['new'] = df[c_col].apply(lambda x: x if x in self.categories_ else 'Other')
		# return self
		return self
	
	def fit_transform(self, df, y=None):
		"""
		fits and transforms the original dataset 'df'
		
		Parameters:
		
		df: dataframe
			original dataset to be transformed
		y: Not used
		
		Returns: a new dataframe containing all desired columns (specified in categories) derived from \\\
		the values that the c_col could take as well as an 'other' column. 
		
		"""

		X = pd.DataFrame(index=df.index)
		X['new'] = df[self.c_col].apply(lambda x: x if x in self.categories else f'{self.c_col}_Other')
		unique = set(X['new'].unique())
		X = pd.get_dummies(X, columns=['new'], prefix='', prefix_sep='').fillna(0)
		# add columns that weren't generated in X that were designated in self.categories.
		for col in list(set(self.categories + [f'{self.c_col}_Other']).difference(unique)):
			X[col] = 0
		return X
	
	
	def transform(self,df):
		
		"""combines the fit and transform methods in one step.
		
		returns:  a new dataframe containing all desired columns (specified in categories) derived from \\\
		the values that the c_col could take as well as an 'other' column. 
		"""
		X = pd.DataFrame(index=df.index)
		X['new'] = df[self.c_col].apply(lambda x: x if x in self.categories else f'{self.c_col}_Other')
		unique = set(X['new'].unique())
		X = pd.get_dummies(X, columns=['new'], prefix='', prefix_sep='').fillna(0)
		# add columns that weren't generated in X that were designated in self.categories.
		for col in list(set(self.categories + [f'{self.c_col}_Other']).difference(unique)):
			X[col] = 0
		return X
		
	def set_output(self,transform):
		pass
 
	
	
class SelectiveScaler(BaseEstimator, TransformerMixin):
	"""Class providing greater control over which columns to perform scaling on (i.e. excludes dummy columns)"""
	def __init__(self, scaler):
		""" Parameters:
			scaler: an Sklearn scaler
		"""
		self.scaler = scaler
	
	def fit(self, X, y=None):
		"""fits the desired scaler using only the columns where x has a maximum value greater than 1
		Parameters:
		X: dataframe
			the original dataset containing the columns requiring scaling
		y: not used
		
		Returns: fitted instance of object
		"""
		
		self.columns_to_scale_ = [c for c in X.columns if X.describe().loc['max',c].max() > 1]
		# Fit the scaler on the selected columns
		self.scaler.fit(X[self.columns_to_scale_])
		return self
	
	def transform(self, X):
		""" Scales data based on the scaler provided at instantiation
		
		Parameters:
		X: pandas dataframe
			The original dataset to scale
		Returns: dataframe containing the scaled data
		"""
		
		X_scaled = X.copy()
		X_scaled = self.scaler.transform(X[self.columns_to_scale_])
		no_scale = [c for c in X.columns if c not in self.columns_to_scale_]
		X_scaled = pd.concat([X_scaled,X[no_scale]],axis=1)
		return X_scaled
	
	def set_output(transform):
		pass