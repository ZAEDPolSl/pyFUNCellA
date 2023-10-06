/*
 * Code generation for function 'calculate_SW'
 *
 */

/* Include files */
#include "calculate_SW.h"
#include "fetch_thresholds.h"
#include "fetch_thresholds_emxutil.h"
#include "fetch_thresholds_rtwutil.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * function [ SW ] = calculate_SW( x, KS )
 */
double calculate_SW(const emxArray_real_T *x, double KS)
{
  double b_ex;
  double d;
  double ex;
  int idx;
  int k;
  int n;
  boolean_T exitg1;
  /*      SW = iqr(x)/(10*KS); */
  /* 'calculate_SW:3' SW = ((max(x)-min(x))/(4*KS))^2; */
  n = x->size[0];
  if (x->size[0] <= 2)
  {
    if (x->size[0] == 0)
    {
      ex = rtNaN;
    }
    else if (x->size[0] == 1)
    {
      ex = x->data[0];
    }
    else if ((x->data[0] < x->data[1]) || (rtIsNaN(x->data[0]) && (!rtIsNaN(x->data[1]))))
    {
      ex = x->data[1];
    }
    else
    {
      ex = x->data[0];
    }
  }
  else
  {
    if (!rtIsNaN(x->data[0]))
    {
      idx = 1;
    }
    else
    {
      idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= x->size[0]))
      {
        if (!rtIsNaN(x->data[k - 1]))
        {
          idx = k;
          exitg1 = true;
        }
        else
        {
          k++;
        }
      }
    }

    if (idx == 0)
    {
      ex = x->data[0];
    }
    else
    {
      ex = x->data[idx - 1];
      idx++;
      for (k = idx; k <= n; k++)
      {
        d = x->data[k - 1];
        if (ex < d)
        {
          ex = d;
        }
      }
    }
  }

  n = x->size[0];
  if (x->size[0] <= 2)
  {
    if (x->size[0] == 0)
    {
      b_ex = rtNaN;
    }
    else if (x->size[0] == 1)
    {
      b_ex = x->data[0];
    }
    else if ((x->data[0] > x->data[1]) || (rtIsNaN(x->data[0]) && (!rtIsNaN(x->data[1]))))
    {
      b_ex = x->data[1];
    }
    else
    {
      b_ex = x->data[0];
    }
  }
  else
  {
    if (!rtIsNaN(x->data[0]))
    {
      idx = 1;
    }
    else
    {
      idx = 0;
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= x->size[0]))
      {
        if (!rtIsNaN(x->data[k - 1]))
        {
          idx = k;
          exitg1 = true;
        }
        else
        {
          k++;
        }
      }
    }

    if (idx == 0)
    {
      b_ex = x->data[0];
    }
    else
    {
      b_ex = x->data[idx - 1];
      idx++;
      for (k = idx; k <= n; k++)
      {
        d = x->data[k - 1];
        if (b_ex > d)
        {
          b_ex = d;
        }
      }
    }
  }

  ex = (ex - b_ex) / (4.0 * KS);
  return ex * ex;
}

/* End of code generation (calculate_SW.c) */
