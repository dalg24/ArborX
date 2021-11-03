/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_PERMUTED_DATA_HPP
#define ARBORX_DETAILS_PERMUTED_DATA_HPP

#include <ArborX_AccessTraits.hpp>

namespace ArborX
{

namespace Details
{

enum class Attachment
{
  none,
  permuted_index,
  original_index
};

template <typename Data, typename Permute, Attachment = Attachment::none>
struct PermutedData
{
  Data _data;
  Permute _permute;
  // NOTE see if const-correctness must be sacrificed here
  // KOKKOS_FUNCTION auto &operator()(int i) { return _data(_permute(i)); }
  KOKKOS_FUNCTION auto /*const*/ &operator()(int i) const
  {
    return _data(_permute(i));
  }
};

} // namespace Details

template <typename Predicates, typename Permute,
          Details::Attachment attachment_kind>
struct AccessTraits<Details::PermutedData<Predicates, Permute, attachment_kind>,
                    PredicatesTag>
{
  using PermutedPredicates =
      Details::PermutedData<Predicates, Permute, attachment_kind>;
  using NativeAccess = AccessTraits<Predicates, PredicatesTag>;
  using Attachment = Details::Attachment;
  using size_type = std::size_t;
  using memory_space = typename NativeAccess::memory_space;

  static KOKKOS_FUNCTION size_type size(PermutedPredicates const &x)
  {
    return NativeAccess::size(x._data);
  }

  template <Attachment dummy = attachment_kind,
            std::enable_if_t<attachment_kind == dummy &&
                             attachment_kind == Attachment::original_index> * =
                nullptr>
  static KOKKOS_FUNCTION auto get(PermutedPredicates const &x, size_type index)
  {
    auto const permuted_index = x._permute(index);
    return attach(NativeAccess::get(x._data, permuted_index), (int)index);
  }

  template <Attachment dummy = attachment_kind,
            std::enable_if_t<attachment_kind == dummy &&
                             attachment_kind == Attachment::permuted_index> * =
                nullptr>
  static KOKKOS_FUNCTION auto get(PermutedPredicates const &x, size_type index)
  {
    auto const permuted_index = x._permute(index);
    return attach(NativeAccess::get(x._data, permuted_index),
                  (int)permuted_index);
  }

  template <Attachment dummy = attachment_kind,
            std::enable_if_t<attachment_kind == dummy &&
                             attachment_kind == Attachment::none> * = nullptr>
  static KOKKOS_FUNCTION auto get(PermutedPredicates const &x, size_type index)
  {
    auto const permuted_index = x._permute(index);
    return NativeAccess::get(x._data, permuted_index);
  }
};

} // namespace ArborX

#endif
