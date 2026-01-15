# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for deprecation utilities."""

import warnings
import pytest

from nemo_rl.compat.deprecation import (
    deprecated_import,
    deprecated_function,
    deprecated_class,
    deprecated_method,
    deprecated_parameter,
    DeprecatedAlias,
    MIGRATION_GUIDE_URL,
)


class TestDeprecatedImport:
    """Tests for deprecated_import function."""

    def test_emits_deprecation_warning(self):
        """Test that deprecated_import emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated_import("OldName", "NewName", "new.module")
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OldName" in str(w[0].message)
            assert "NewName" in str(w[0].message)
            assert "new.module" in str(w[0].message)

    def test_includes_migration_guide_url(self):
        """Test that warning includes migration guide URL."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated_import("Old", "New", "module")
            
            assert MIGRATION_GUIDE_URL in str(w[0].message)


class TestDeprecatedFunction:
    """Tests for deprecated_function decorator."""

    def test_decorated_function_warns(self):
        """Test that decorated function emits warning."""
        @deprecated_function("new_func", "new.module")
        def old_func():
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            
            assert result == "result"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func" in str(w[0].message)
            assert "new_func" in str(w[0].message)

    def test_preserves_function_name(self):
        """Test that decorator preserves function name."""
        @deprecated_function("new", "module")
        def my_function():
            """My docstring."""
            pass
        
        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestDeprecatedClass:
    """Tests for deprecated_class decorator."""

    def test_warns_on_instantiation(self):
        """Test that warning is emitted when class is instantiated."""
        @deprecated_class("NewClass", "new.module")
        class OldClass:
            def __init__(self, value):
                self.value = value
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            instance = OldClass(42)
            
            assert instance.value == 42
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OldClass" in str(w[0].message)
            assert "NewClass" in str(w[0].message)


class TestDeprecatedMethod:
    """Tests for deprecated_method decorator."""

    def test_warns_on_method_call(self):
        """Test that warning is emitted when method is called."""
        class MyClass:
            @deprecated_method("new_method")
            def old_method(self):
                return "result"
        
        obj = MyClass()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_method()
            
            assert result == "result"
            assert len(w) == 1
            assert "old_method" in str(w[0].message)
            assert "new_method" in str(w[0].message)


class TestDeprecatedParameter:
    """Tests for deprecated_parameter function."""

    def test_warns_for_renamed_param(self):
        """Test warning for renamed parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated_parameter("old_param", "new_param")
            
            assert len(w) == 1
            assert "old_param" in str(w[0].message)
            assert "new_param" in str(w[0].message)

    def test_warns_for_removed_param(self):
        """Test warning for removed parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            deprecated_parameter("removed_param", None)
            
            assert len(w) == 1
            assert "removed_param" in str(w[0].message)


class TestDeprecatedAlias:
    """Tests for DeprecatedAlias class."""

    def test_warns_on_call(self):
        """Test that warning is emitted when alias is called."""
        def target_func(x):
            return x * 2
        
        alias = DeprecatedAlias(target_func, "old_name", "new_name", "new.module")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = alias(5)
            
            assert result == 10
            assert len(w) == 1
            assert "old_name" in str(w[0].message)
            assert "new_name" in str(w[0].message)

    def test_warns_on_attribute_access(self):
        """Test that warning is emitted on attribute access."""
        class Target:
            attr = "value"
        
        alias = DeprecatedAlias(Target, "OldTarget", "NewTarget", "new.module")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = alias.attr
            
            assert value == "value"
            assert len(w) == 1
